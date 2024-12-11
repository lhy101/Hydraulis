import os
import signal
import time
import argparse
import ast
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from data_utils import LLaMAJsonDataset, build_data_loader
import seaborn as sns

# 使用 seaborn 的美化主题
sns.set_theme(style="whitegrid")

dataset_1 = "web"
dataset_2 = "code"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dataset = "commoncrawl"
dataset = "web"
# dataset = "github"
# dataset = "code"
counter_file_path = f"./dataset_analysis/64k_{dataset}_counter.pkl"
cdf_file_path = f"./dataset_analysis/{dataset}_cdf.png"
max_counter_file_path = f"./dataset_analysis/{dataset}_max_counter.pkl"
max_cdf_file_path = f"./dataset_analysis/{dataset}_max_cdf.png"
simulation_file_path = f"./dataset_analysis/{dataset}_simulation.png"

def read_counter():
    with open(counter_file_path, 'rb') as file:
        counter = pickle.load(file)
    print(f"Max seq len is {max(counter.keys())}")

def scan_and_dump(dataset, batch_size=1000):
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # Initialize an empty list to store sequence lengths
    seqlen_list = []
    # Process data in batches
    for i in tqdm(range(0, total_seqs, batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        seqlen_list.extend(batch_seqlen)
    # Convert the list to a numpy array for further processing
    # Count the occurrences of each sequence length
    counter = Counter(seqlen_list)
    with open(counter_file_path, 'wb') as file:
        pickle.dump(counter, file)
    x_vals, counts = zip(*sorted(counter.items()))  
    # Calculate the cumulative distribution function (CDF)
    y_vals = np.cumsum(counts) / total_seqs
    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='CDF', color='blue', lw=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.3)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(cdf_file_path)
    plt.show()
    
def scan_and_dump_max_seqlen(dataset, batch_size=64):
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # Initialize an empty list to store sequence lengths
    max_seqlen_list = []
    # Process data in batches
    for i in tqdm(range(0, total_seqs, batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        max_seqlen_list.append(max(batch_seqlen))
    # Convert the list to a numpy array for further processing
    # Count the occurrences of each sequence length
    counter = Counter(max_seqlen_list)
    with open(max_counter_file_path, 'wb') as file:
        pickle.dump(counter, file)
    x_vals, counts = zip(*sorted(counter.items()))  
    # Calculate the cumulative distribution function (CDF)
    y_vals = np.cumsum(counts) / len(max_seqlen_list)
    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='CDF', color='blue', lw=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.3)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(max_cdf_file_path)
    plt.show()

def draw_sample_simulation(dataset, batch_size=64):
    # 假设 dataset 和 batch_size 已经定义
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # 初始化列表以存储每个batch的最大序列长度和每个batch中所有序列的长度
    max_seqlen_list = []
    batch_indices = []
    all_seqlen_list = []
    # 处理数据
    for i in tqdm(range(0, min(total_seqs, 100 * batch_size), batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        # 计算每个序列的长度，假设pad_id是用来填充的标记
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        # 记录当前batch的最大序列长度
        max_seqlen_list.append(max(batch_seqlen))
        # 记录当前batch的所有序列长度
        all_seqlen_list.extend(batch_seqlen)
        # 为每个序列分配相同的批次索引
        batch_indices.extend([i // batch_size] * len(batch_seqlen))
    # 绘制图像
    plt.figure(figsize=(12, 6))
    # 绘制最大序列长度的折线图
    plt.plot(range(len(max_seqlen_list)), max_seqlen_list, label='Max Sequence Length', color='blue', marker='o', markersize=4)
    # 绘制所有序列长度的散点图，使用相同的 batch 索引
    plt.scatter(batch_indices, all_seqlen_list, color='r', label='Sequence Length', alpha=0.5, s=10)
    # 添加标题和标签
    plt.title('Sequence Length Distribution of Different Batches')
    plt.xlabel('Batch Index')
    # 添加图例
    plt.legend()
    # 美化图形
    plt.grid(True, linestyle='--', alpha=0.6)
    # 展示图像
    plt.tight_layout()
    plt.savefig(simulation_file_path)
    plt.show()
    
def draw_sample_simulation(datasets, labels, global_batch_size=None, global_token_num=100000, simulation_file_path="simulation"):
    # 创建 1x4 的子图布局，宽度比例为 3:1:3:1
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [3, 1, 3, 1]}, sharex='col')

    # 定义柱状图的区间边界
    bins = [0, 1024, 2048, 4096, 8192, 16384, 10000000000]
    bin_labels = ['0-1k', '1k-2k', '2k-4k', '4k-8k', '8k-16k', '16k-32k']

    # 遍历每个数据集
    for idx, (dataset, label) in enumerate(zip(datasets, labels)):
        data = dataset.data
        total_seqs = len(data)
        print(f"Total seqs in dataset {label}: {total_seqs}")
        
        # 初始化列表以存储每个batch的最大序列长度和每个batch中所有序列的长度
        max_seqlen_list = []
        batch_indices = []
        all_seqlen_list = []
        
        # 处理数据
        dataloader = build_data_loader(dataset, 0, global_batch_size=global_batch_size, global_token_num=global_token_num)
        train_data_iter = iter(dataloader)
        batch_data_list = []
        for i in tqdm(range(100), desc=f"Processing batches for {label}"):
            try:
                batch_data = next(train_data_iter)
            except StopIteration:
                break
            batch_data_list.append(batch_data)
            batch_array = np.array(batch_data)
            # 计算每个序列的长度，假设pad_id是用来填充的标记
            batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
            # 记录当前batch的最大序列长度
            max_seqlen_list.append(max(batch_seqlen))
            # 记录当前batch的所有序列长度
            all_seqlen_list.extend(batch_seqlen)
            # 为每个序列分配相同的批次索引
            batch_indices.extend([i] * len(batch_data))
        
        # 绘制最大序列长度的折线图
        # axs[2*idx].plot(range(len(max_seqlen_list)), max_seqlen_list, label=f'Max Sequence Length', color='#1f77b4', marker='o', markersize=4, linestyle='--', linewidth=2)
        axs[2*idx].plot(range(len(max_seqlen_list)), max_seqlen_list, label=f'Max Sequence Length', color='r', marker='o', alpha=0.5, markersize=4, linestyle='--', linewidth=1.5)
        # 绘制所有序列长度的散点图，使用相同的 batch 索引
        axs[2*idx].scatter(batch_indices, all_seqlen_list, color='r', label=f'Sequence Length', 
                            alpha=0.5, s=10)
        
        # 添加标题并加粗
        axs[2*idx].set_title(f'{label}', fontweight='bold', fontsize=14)
        # 添加网格线
        axs[2*idx].grid(True, linestyle='--', alpha=0.6)

        # 处理倒数第二个 batch 的序列长度分布
        last_batch_data = batch_data_list[-1]
        last_batch_array = np.array(last_batch_data)
        last_batch_seqlen = np.sum(last_batch_array != dataset.encoder.pad_id(), axis=1)

        # 统计最后一个 batch 中各个区间的序列数量
        hist, _ = np.histogram(last_batch_seqlen, bins=bins)
        print(hist)

        # 绘制直方图
        axs[2*idx + 1].bar(range(len(bin_labels)), hist, color='gray', alpha=0.6, width=0.6)
        
        # 设置柱状图的 X 轴范围和标签
        axs[2*idx + 1].set_xticks(range(len(bin_labels)))
        axs[2*idx + 1].set_title('Count', fontsize=12)
        axs[2*idx + 1].set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=10)

        
        # 设置右侧Y轴标签
        # axs[2*idx + 1].set_ylabel('Sequence Count', fontsize=12)
    
    # 公用X轴标签
    axs[0].set_xlabel('Iteration', fontsize=12)
    axs[2].set_xlabel('Iteration', fontsize=12)
    # axs[3].set_ylim(0, 25)
    axs[0].set_ylabel('Sequence Length', fontsize=12)
    plt.subplots_adjust(wspace=0.1)  # 调整wspace来控制水平间距，值越小间距越小
    
    # 获取图例句柄和标签（从第一个子图中获取）
    handles, labels = axs[0].get_legend_handles_labels()

    # 添加全局图例，放在图的正下方
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=12, fancybox=True)

    # 调整子图布局
    plt.tight_layout()  

    # 保存并展示图像
    plt.savefig(f"{simulation_file_path}.png", format="png", pad_inches=0.01, bbox_inches="tight")
    plt.savefig(f"{simulation_file_path}.svg", format="svg", pad_inches=0.01, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_seq_len", type=int, default=32768, help="maximum sequence len"
    )
    parser.add_argument(
        "--json_file", type=str, default=f"data/data.json", help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, default="content", help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, default="data/vocab.json", help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, default="data/merges.txt", help='gpt merge file path'
    )
    args = parser.parse_args()
    dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file
    )
    scan_and_dump(dataset)
    # scan_and_dump_max_seqlen(dataset)
    # read_counter()
    # draw_sample_simulation(dataset)
    # draw_sample_simulation_new(datasets=[dataset_1, dataset_2], labels=["CommonCrawl", "GitHub"], simulation_file_path="combine_simulation")