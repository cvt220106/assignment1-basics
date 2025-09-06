use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

/// [RUST IMPLEMENTATION]
/// Performs the BPE merge on the word frequency counter and calculates the
/// change in pair statistics in a single pass. This is the high-performance
/// replacement for the Python version.
///
/// Args:
///     word_counts (dict): The current word frequency counter.
///     pair_to_merge (tuple): The byte pair to be merged (e.g., (116, 104) for 'th').
///     new_token_id (int): The ID of the new token created from the merge.
///
/// Returns:
///     A tuple containing:
///     - new_word_counts (dict): The updated word frequency counter after the merge.
///     - stats_delta (dict): A dictionary representing the changes to the pair statistics.
#[pyfunction]
fn merge_word_counts_and_update_stats_rust(
    word_counts: &PyDict,
    pair_to_merge: (u32, u32),
    new_token_id: u32,
) -> PyResult<PyObject> {
    // PyO3 提供的 Python 全局锁，确保线程安全
    Python::with_gil(|py| {
        // 创建 Rust 的 HashMap 来存储结果
        let mut new_rust_word_counts: HashMap<Vec<u32>, isize> = HashMap::new();
        let mut stats_delta: HashMap<(u32, u32), isize> = HashMap::new();

        let (p1, p2) = pair_to_merge;

        // 遍历从 Python 传来的字典
        for (word_py, count_py) in word_counts.iter() {
            // 将 Python 的 key (tuple) 和 value (int) 转换成 Rust 类型
            let word: Vec<u32> = word_py.extract()?;
            let count: isize = count_py.extract()?;

            if word.len() < 2 {
                // 如果单词太短，不可能发生合并，直接加入新集合
                *new_rust_word_counts.entry(word).or_insert(0) += count;
                continue;
            }

            let mut new_word: Vec<u32> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && (word[i], word[i + 1]) == (p1, p2) {
                    // 找到合并点，更新统计变化量

                    // 1. 减去被破坏的旧词对的频率
                    if i > 0 {
                        *stats_delta.entry((word[i - 1], p1)).or_insert(0) -= count;
                    }
                    if i < word.len() - 2 {
                        *stats_delta.entry((p2, word[i + 2])).or_insert(0) -= count;
                    }

                    // 2. 加上新产生的词对的频率
                    if i > 0 {
                        *stats_delta.entry((word[i - 1], new_token_id)).or_insert(0) += count;
                    }
                    if i < word.len() - 2 {
                        *stats_delta.entry((new_token_id, word[i + 2])).or_insert(0) += count;
                    }

                    new_word.push(new_token_id);
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }
            // 将合并后的新词加入新的 word_counts
            *new_rust_word_counts.entry(new_word).or_insert(0) += count;
        }

        // 将 Rust 的 HashMap 转换回 Python 的字典以便返回
        let py_new_word_counts = PyDict::new(py);
        for (word, count) in new_rust_word_counts.iter() {
            py_new_word_counts.set_item(PyTuple::new(py, word), count)?;
        }

        let py_stats_delta = PyDict::new(py);
        for (pair, delta) in stats_delta.iter() {
            py_stats_delta.set_item(pair, delta)?;
        }

        // 将两个字典打包成一个 Python 元组返回
        Ok(PyTuple::new(py, &[py_new_word_counts, py_stats_delta]).to_object(py))
    })
}


/// 这个宏定义了 Python 模块的入口点。
/// 当你在 Python 中 `import rust_bpe_optimizer` 时，这个函数会被调用。
#[pymodule]
fn rust_bpe_optimizer(_py: Python, m: &PyModule) -> PyResult<()> {
    // 将我们的 Rust 函数添加到 Python 模块中，这样就可以在 Python 里调用它了
    m.add_function(wrap_pyfunction!(merge_word_counts_and_update_stats_rust, m)?)?;
    Ok(())
}