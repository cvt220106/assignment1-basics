import collections
import regex as re
import multiprocessing
import os
from typing import BinaryIO, Counter as CounterType, Tuple, Dict, List

try:
    from rust_bpe_optimizer import merge_word_counts_and_update_stats_rust
    RUST_AVAILABLE = True
    print("🚀 Successfully imported Rust optimizer!")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️ Could not import Rust optimizer. Falling back to pure Python version.")

# GPT-2 的预分词正则表达式
# 详细解释:
# 's|'t|'re|'ve|'m|'ll|'d : 匹配常见的缩写
# ?\p{L}+ : 匹配一个或多个 Unicode 字母 (例如 "hello", "你好")
# ?\p{N}+ : 匹配一个或多个 Unicode 数字 (例如 "123")
# ?[^\s\p{L}\p{N}]+ : 匹配一个或多个非空格、非字母、非数字的字符 (例如 ".,?!")
# \s+(?!\S) : 匹配一个或多个空格，但只在它后面没有非空格字符时 (处理行尾空格)
# \s+ : 匹配一个或多个空格
GPT2_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    函数说明:
        在一个大文件中找到安全的分块边界，以创建期望数量的块。
        每个边界都位于指定特殊标记的开头，以确保逻辑单元（如文档）不会被切分。
        此函数为并行处理中的负载均衡进行了优化。

    输入:
        file (BinaryIO): 用于读取的二进制文件句柄。
        desired_num_chunks (int): 目标块数量。实际数量可能因标记分布而减少。
        split_special_token (bytes): 标记安全边界的特殊标记（字节形式），例如 b"<|endoftext|>"。

    输出:
        list[int]: 一个排序后的文件内字节偏移量列表，代表每个块的开始和结束。
                   列表包含 0 和文件总大小。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0, 0]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks)]
    chunk_boundaries.append(file_size)

    # Refine boundaries to align with the special token
    mini_chunk_size = 4096  # Read 4KB at a time to find the token
    for i in range(1, len(chunk_boundaries) - 1):
        position = chunk_boundaries[i]
        file.seek(position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                position = file_size  # Reached end of file
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                position += found_at
                break
            position += len(mini_chunk)
        chunk_boundaries[i] = position

    # Add the start boundary and ensure uniqueness and order
    final_boundaries = sorted(list(set([0] + chunk_boundaries)))
    return final_boundaries


def process_chunk_for_counts(
        chunk_data: tuple[bytes, str, re.Pattern]
) -> CounterType[tuple[int, ...]]:
    """
    函数说明:
        一个用于多进程的工作函数，处理单个文本块。
        它首先根据特殊标记分割文本块，然后用正则表达式对子块进行预分词，
        最后返回每个独一无二的“单词”的频率计数。

    输入:
        chunk_data (tuple): 一个元组，包含原始文本块字节、用于分割特殊标记的
                            正则表达式，以及 BPE 预分词的正则表达式。

    输出:
        CounterType[tuple[int, ...]]: 一个计数器，键是预分词后的“单词”（以字节值的元组表示），
                                      值是它们的出现频率。
    """
    chunk_bytes, special_tokens_pattern, regex_pattern = chunk_data
    text = chunk_bytes.decode('utf-8', errors='replace')

    word_counts = collections.Counter()
    # First, split the chunk by ALL special tokens to prevent merges across them
    sub_chunks = re.split(special_tokens_pattern, text)

    for sub_chunk in sub_chunks:
        if sub_chunk:
            for match in regex_pattern.finditer(sub_chunk):
                word_bytes = match.group(0).encode('utf-8')
                word_counts[tuple(word_bytes)] += 1
    return word_counts


def get_stats_from_word_counts(
        word_counts: CounterType[tuple[int, ...]]
) -> CounterType[tuple[int, int]]:
    """
    函数说明:
        从“单词”频率计数器中高效地计算所有相邻字节对的初始频率。
        此方法操作的是唯一词，并通过乘以其频率来计算，远比遍历整个语料库高效。

    输入:
        word_counts (CounterType): 唯一词及其频率的计数器。

    输出:
        CounterType[tuple[int, int]]: 相邻字节对及其在整个语料库中总频率的计数器。
    """
    pair_stats = collections.Counter()
    for word_tuple, count in word_counts.items():
        for pair in zip(word_tuple[:-1], word_tuple[1:]):
            pair_stats[pair] += count
    return pair_stats


def merge_word_counts_and_update_stats(
        word_counts: CounterType[Tuple[int, ...]],
        pair_to_merge: Tuple[int, int],
        new_token_id: int
) -> Tuple[CounterType[Tuple[int, ...]], CounterType[Tuple[int, int]]]:
    """
    函数说明:
        在词频计数器上执行 BPE 合并操作，并一步计算出词对频率的变化。

    输入:
        word_counts (CounterType): 当前的词频计数器。
        pair_to_merge (Tuple[int, int]): 需要被合并的字节对（例如 (116, 104) 代表 'th'）。
        new_token_id (int): 合并后产生的新词元 ID。

    输出:
        Tuple[CounterType, CounterType]: 一个元组，包含：
        - new_word_counts: 合并后更新的词频计数器。
        - stats_delta: 一个代表词对频率变化量（增量）的计数器，供主循环应用。
    """

    new_word_counts = collections.Counter()
    stats_delta = collections.Counter()
    p1, p2 = pair_to_merge

    for word, count in word_counts.items():
        if len(word) < 2:
            new_word_counts[word] += count
            continue

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
                # This is the merge point. Update stats based on neighbors in the ORIGINAL word.
                # 1. Decrement counts for pairs being destroyed.
                if i > 0:
                    stats_delta[(word[i - 1], p1)] -= count
                if i < len(word) - 2:
                    stats_delta[(p2, word[i + 2])] -= count

                # 2. Increment counts for new pairs being formed.
                if i > 0:
                    stats_delta[(word[i - 1], new_token_id)] += count
                if i < len(word) - 2:
                    stats_delta[(new_token_id, word[i + 2])] += count

                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_tuple = tuple(new_word)
        new_word_counts[new_word_tuple] += count

    return new_word_counts, stats_delta


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    函数说明:
        从一个文本文件训练一个字节对编码（BPE）分词器。
        此实现经过高度优化，使用了并行预分词和基于词频的高效合并策略。

    输入:
        input_path (str): 训练文本文件的路径。
        vocab_size (int): 期望的最终词汇表大小。
        special_tokens (list[str]): 需要添加到词汇表中的特殊标记列表。
                                    第一个特殊标记将被用作文档分隔符以进行并行处理。

    输出:
        tuple: 一个元组，包含：
        - vocab (dict[int, bytes]): 最终的词汇表，映射 token ID 到字节。
        - merges (list[tuple[bytes, bytes]]): 按创建顺序列出的合并规则列表。
    """
    assert vocab_size >= 256 + len(special_tokens), "Vocab size is too small."

    # --- 1. Vocabulary Initialization ---
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode('utf-8')

    # --- 2. Parallel Pre-tokenization ---
    print("Starting parallel pre-tokenization to get word counts...")
    special_tokens_pattern = "|".join(re.escape(st) for st in special_tokens)
    doc_end_token = b"<|endoftext|>"
    num_processes = 16

    chunks_data = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, doc_end_token)
        print(f"File split into {len(boundaries) - 1} chunks for {num_processes} processes.")
        for start, end in zip(boundaries, boundaries[1:]):
            if start < end:
                f.seek(start)
                chunks_data.append((f.read(end - start), special_tokens_pattern, GPT2_PATTERN))

    word_counts = collections.Counter()
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk_for_counts, chunks_data)
        for res in results:
            word_counts.update(res)

    print(f"Pre-tokenization complete. Found {len(word_counts)} unique 'words'.")

    # --- 3. BPE Training Loop (Optimized) ---
    merges: List[Tuple[bytes, bytes]] = []
    stats = get_stats_from_word_counts(word_counts)

    num_merges_needed = vocab_size - len(vocab)
    for i in range(num_merges_needed):
        if not stats:
            print("No more pairs to merge. Stopping early.")
            break

        max_freq = max(stats.values())

        if max_freq < 1:
            print("Highest pair count is 0. Stopping early.")
            break

        tied_pairs = [p for p, freq in stats.items() if freq == max_freq]
        best_pair = max(tied_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        new_token_id = len(vocab)
        token1_bytes = vocab[best_pair[0]]
        token2_bytes = vocab[best_pair[1]]
        vocab[new_token_id] = token1_bytes + token2_bytes
        merges.append((token1_bytes, token2_bytes))

        if RUST_AVAILABLE:
            # 调用 Rust 函数，它返回的是 (dict, dict)
            new_counts_dict, delta_dict = merge_word_counts_and_update_stats_rust(
                word_counts, best_pair, new_token_id
            )

            # 我们需要将返回的 dict 转换为 Counter
            word_counts = collections.Counter(new_counts_dict)
            stats_delta = collections.Counter(delta_dict)
        else:
            # 调用纯 Python 函数
            word_counts, stats_delta = merge_word_counts_and_update_stats(
                word_counts, best_pair, new_token_id
            )

        stats.update(stats_delta)
        del stats[best_pair]

        if (i + 1) % 100 == 0:
            print(
                f"Merge {i + 1}/{num_merges_needed}: {best_pair} -> {new_token_id} ({vocab[new_token_id]!r}) | Freq: {stats[max(stats, key=stats.get)] if stats else 0}")

    return vocab, merges


if __name__ == '__main__':
    # This block allows the script to be run directly for a demonstration.

    # 1. Create a dummy data file for training.
#     sample_text = """Hello world.\nThis is a sample text for BPE training.\nIt repeats some words, like sample text.
# <|endoftext|>
# This is the second document. It also has sample text.
# The BPE algorithm will learn to merge common character pairs.
# For example, 's' and 'a', then 'sa' and 'm', and so on.
# <|endoftext|>
# The third document is here to provide more data."""
#     sample_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest and the vocabulary has a special token <|endoftext|>"
#
#     input_file_path = "data.txt"
#     with open(input_file_path, "w", encoding="utf-8") as f:
#         f.write(sample_text)
#
#     print(f"Created sample training data at '{input_file_path}'")

    input_file_path = "../data/TinyStoriesV2-GPT4-train.txt"

    # 2. Define training parameters.
    TARGET_VOCAB_SIZE = 1000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    print("\n--- Starting BPE Training ---")

    # 3. Run the training function.
    final_vocab, final_merges = train_bpe(
        input_path=input_file_path,
        vocab_size=TARGET_VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    # 4. Print the results.
    print("\n--- BPE Training Complete ---")

    print(f"\nFinal Vocabulary Size: {len(final_vocab)}")
    print("--- First 10 and all new tokens ---")
    for token_id, token_bytes in sorted(final_vocab.items()):
        if token_id < 10 or token_id >= 256:
            try:
                # Attempt to decode for readability, show bytes on failure
                decoded = token_bytes.decode('utf-8')
                print(f"{token_id}: {decoded!r}  ({token_bytes})")
            except UnicodeDecodeError:
                print(f"{token_id}: {token_bytes}")

    print("\n--- Merge Rules ---")
    for i, pair in enumerate(final_merges):
        p1_decoded = pair[0].decode('utf-8', errors='replace')
        p2_decoded = pair[1].decode('utf-8', errors='replace')
        print(f"{i + 1:02d}: {p1_decoded!r} + {p2_decoded!r} -> {(p1_decoded + p2_decoded)!r}")