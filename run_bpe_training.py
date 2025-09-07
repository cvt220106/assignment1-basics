import time
import json
import os
import psutil
from typing import Dict, List, Tuple

# 假设您的 bpe_tokenizer 模块位于 cs336_basics 包中
from cs336_basics.bpe_tokenizer import train_bpe

# --- 1. 配置 ---
# !!! 重要: 请将此路径修改为您本地 TinyStories 数据集的实际路径 !!!
DATASET_PATH = 'data/TinyStoriesV2-GPT4-train.txt'
VOCAB_SIZE = 10000
# DATASET_PATH = 'data/owt_train.txt'
# VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]

# 输出文件路径
OUTPUT_DIR = "bpe_training_results"
PREFIX = DATASET_PATH.split("/")[-1].split(".")[0]
OUTPUT_VOCAB_PATH = os.path.join(OUTPUT_DIR, f"{PREFIX}_vocab.json")
OUTPUT_MERGES_PATH = os.path.join(OUTPUT_DIR, "merges.json")


def serialize_vocab(vocab: Dict[int, bytes]) -> Dict[str, List[int]]:
    """将词汇表 (dict[int, bytes]) 转换为可序列化为 JSON 的格式 (dict[str, list[int]])。"""
    return {str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()}


def serialize_merges(merges: List[Tuple[bytes, bytes]]) -> List[List[List[int]]]:
    """将合并规则 (list[tuple[bytes, bytes]]) 转换为可序列化为 JSON 的格式。"""
    return [[list(p1), list(p2)] for p1, p2 in merges]


def main():
    """执行 BPE 训练、性能测量、结果保存和分析的主函数。"""
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 找不到数据集文件 '{DATASET_PATH}'。")
        print("请下载 TinyStories 数据集并更新脚本中的 DATASET_PATH 变量。")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- 开始 BPE 训练 ---")
    print(f"数据集: {DATASET_PATH}")
    print(f"目标词汇量大小: {VOCAB_SIZE}")
    print("-" * 25)

    # --- 性能测量 ---
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()

    # --- 执行训练 ---
    final_vocab, final_merges = train_bpe(
        input_path=DATASET_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    end_time = time.perf_counter()

    # 获取内存使用情况 (RSS: Resident Set Size)
    mem_info = process.memory_info()
    memory_used_gb = mem_info.rss / (1024 ** 3)
    duration_minutes = (end_time - start_time) / 60

    print("\n--- 训练完成 ---")
    print(f"训练耗时: {duration_minutes:.2f} 分钟")
    print(f"峰值内存占用: {memory_used_gb:.2f} GB")

    # --- 结果序列化 ---
    print(f"正在将词汇表保存到: {OUTPUT_VOCAB_PATH}")
    serializable_vocab = serialize_vocab(final_vocab)
    with open(OUTPUT_VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=2)

    print(f"正在将合并规则保存到: {OUTPUT_MERGES_PATH}")
    serializable_merges = serialize_merges(final_merges)
    with open(OUTPUT_MERGES_PATH, 'w', encoding='utf-8') as f:
        json.dump(serializable_merges, f)
    print("结果保存完毕。")

    # --- 结果分析 ---
    if final_vocab:
        longest_token_bytes = max(final_vocab.values(), key=len)
        longest_token_len = len(longest_token_bytes)

        try:
            # 尝试将最长的 token 解码为字符串以便查看
            longest_token_str = longest_token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            longest_token_str = str(longest_token_bytes)  # 如果解码失败，则显示原始字节

        print("\n--- 结果分析 ---")
        print(f"词汇表中最长词元的长度: {longest_token_len} 字节")
        print(f"最长的词元: {longest_token_str!r}")


if __name__ == '__main__':
    main()
