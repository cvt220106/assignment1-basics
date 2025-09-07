use chrono::Timelike;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant}; // 用于获取当前时间的小时、分钟、秒

type Word = Vec<u32>;
type Pair = (u32, u32);

// BpeTrainer 结构体保持不变，它封装了训练的核心状态。
#[pyclass]
struct BpeTrainer {
    word_counts: HashMap<Word, isize>,
    stats: HashMap<Pair, isize>,
    pair_to_words_index: HashMap<Pair, HashSet<Word>>,
}

#[pymethods]
impl BpeTrainer {
    // 构造函数现在也接受初始词汇表，用于初始化倒排索引。
    #[new]
    fn new(py_word_counts: &PyDict) -> PyResult<Self> {
        let word_counts: HashMap<Word, isize> = py_word_counts.extract()?;
        let mut stats = HashMap::new();
        let mut pair_to_words_index: HashMap<Pair, HashSet<Word>> = HashMap::new();

        for (word, &count) in &word_counts {
            if word.len() > 1 {
                for i in 0..(word.len() - 1) {
                    let pair = (word[i], word[i + 1]);
                    *stats.entry(pair).or_insert(0) += count;
                    pair_to_words_index
                        .entry(pair)
                        .or_default()
                        .insert(word.clone());
                }
            }
        }

        Ok(BpeTrainer {
            word_counts,
            stats,
            pair_to_words_index,
        })
    }

    // merge_one 现在接受 vocab 作为参数，以便进行正确的平局决胜。
    fn merge_one(&mut self, vocab: &PyDict) -> PyResult<Option<(PyObject, PyObject, u32)>> {
        Python::with_gil(|py| {
            if self.stats.is_empty() {
                return Ok(None);
            }

            let max_freq = *self.stats.values().max().unwrap_or(&0);
            if max_freq < 1 {
                return Ok(None);
            }

            // 【修正】平局决胜逻辑现在正确地从传入的 vocab 中获取字节
            let best_pair = self
                .stats
                .iter()
                .filter(|(_, &freq)| freq == max_freq)
                .max_by_key(|(pair, _)| {
                    let p1_bytes: Vec<u8> =
                        vocab.get_item(&pair.0).unwrap().unwrap().extract().unwrap();
                    let p2_bytes: Vec<u8> =
                        vocab.get_item(&pair.1).unwrap().unwrap().extract().unwrap();
                    (p1_bytes, p2_bytes)
                })
                .map(|(pair, _)| *pair)
                .unwrap();

            let new_token_id = vocab.len() as u32;

            if let Some(affected_words) = self.pair_to_words_index.remove(&best_pair) {
                for word in affected_words {
                    if let Some(count) = self.word_counts.remove(&word) {
                        self.update_stats_for_word(&word, -count, true);
                        let new_word = self.merge_word(&word, &best_pair, new_token_id);
                        self.update_stats_for_word(&new_word, count, false);
                        *self.word_counts.entry(new_word).or_insert(0) += count;
                    }
                }
            }

            self.stats.remove(&best_pair);

            let p1_bytes = vocab.get_item(best_pair.0)?.unwrap().to_object(py);
            let p2_bytes = vocab.get_item(best_pair.1)?.unwrap().to_object(py);
            Ok(Some((p1_bytes, p2_bytes, new_token_id)))
        })
    }
}

// 内部辅助函数保持不变
impl BpeTrainer {
    fn merge_word(&self, word: &Word, pair: &Pair, new_token_id: u32) -> Word {
        let mut new_word = Vec::with_capacity(word.len());
        let mut i = 0;
        while i < word.len() {
            if i < word.len() - 1 && (word[i], word[i + 1]) == *pair {
                new_word.push(new_token_id);
                i += 2;
            } else {
                new_word.push(word[i]);
                i += 1;
            }
        }
        new_word
    }

    fn update_stats_for_word(&mut self, word: &Word, delta: isize, is_removal: bool) {
        if word.len() < 2 {
            return;
        }
        for i in 0..(word.len() - 1) {
            let pair = (word[i], word[i + 1]);
            *self.stats.entry(pair).or_insert(0) += delta;
            if self.stats.get(&pair).map_or(false, |&v| v <= 0) {
                self.stats.remove(&pair);
            }

            if is_removal {
                if let Some(set) = self.pair_to_words_index.get_mut(&pair) {
                    set.remove(word);
                }
            } else {
                self.pair_to_words_index
                    .entry(pair)
                    .or_default()
                    .insert(word.clone());
            }
        }
    }
}

// 获取当前时间的格式化字符串
fn get_time_string() -> String {
    // 使用当地时间而不是UTC时间
    let now = chrono::Local::now();

    // 格式化为 HH:MM:SS.mmm 格式
    now.format("%H:%M:%S.%3f").to_string();
    // 格式化为 HH:MM:SS.mmm 格式（只保留3位毫秒）
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        now.hour(),
        now.minute(),
        now.second(),
        now.nanosecond() / 1_000_000
    )
}

#[pyfunction]
fn train_bpe_with_rust(
    py_word_counts: &PyDict,
    vocab_size: usize,
    initial_vocab: &PyDict,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 记录开始时间
        let start_time = Instant::now();

        // 打印开始时间
        println!("[{}] Starting BPE training in Rust", get_time_string());

        // 从 Python 传入的 initial_vocab 创建一个可变的副本。
        let vocab = initial_vocab.copy()?;

        let mut trainer = BpeTrainer::new(py_word_counts)?;
        let mut merges = Vec::new();

        let num_merges_needed = vocab_size.saturating_sub(vocab.len());
        println!(
            "[{}] Need to perform {} merges with initial vocab size {}",
            get_time_string(),
            num_merges_needed,
            vocab.len()
        );
        for i in 0..num_merges_needed {
            if let Some((p1_bytes, p2_bytes, new_token_id)) = trainer.merge_one(vocab)? {
                let new_token_bytes_obj =
                    p1_bytes.call_method1(py, "__add__", (p2_bytes.clone_ref(py),))?;
                vocab.set_item(new_token_id, &new_token_bytes_obj)?;
                merges.push((p1_bytes, p2_bytes));

                if (i + 1) % 500 == 0 {
                    // 增加打印频率以更好地观察进度
                    println!(
                        "[{}] [{}s] Rust Merge {}/{}",
                        get_time_string(),
                        start_time.elapsed().as_secs_f32(),
                        i + 1,
                        num_merges_needed
                    );
                }
            } else {
                println!(
                    "[{}] [{}s] No more pairs to merge. Stopping early in Rust.",
                    get_time_string(),
                    start_time.elapsed().as_secs_f32()
                );
                break;
            }
        }

        // 返回的 merges 是 PyList of PyTuples，这与 Python 的 list[tuple] 类型对应。
        let py_merges = PyList::new(
            py,
            merges.iter().map(|(p1, p2)| PyTuple::new(py, &[p1, p2])),
        );
        let vocab = vocab.to_object(py);
        let py_merges = py_merges.to_object(py);
        // 打印总耗时
        println!(
            "[{}] Finished BPE training in Rust. Total time: {:.2}s",
            get_time_string(),
            start_time.elapsed().as_secs_f32()
        );

        Ok(PyTuple::new(py, &[vocab, py_merges]).to_object(py))
    })
}

#[pymodule]
fn rust_bpe_optimizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_with_rust, m)?)?;
    m.add_class::<BpeTrainer>()?; // BpeTrainer 自身不需要导出给 Python，但保留亦无妨。
    Ok(())
}
