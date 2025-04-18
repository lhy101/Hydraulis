import os
import pyarrow.parquet as pq
import pandas as pd

# samples_num = 128000
samples_num = None
only_first = True
root_folder = '/home/pkuhydraulis/lhy/multi_switch/examples/hydraulis/data'
web_folder = os.path.join(root_folder, 'github')

# 初始化一个空的列表来存储所有 DataFrame
dfs = []

# 遍历 web_folder 目录下的所有文件
for file_name in os.listdir(web_folder):
    if file_name.endswith('.parquet'):
        # 读取每个 parquet 文件
        file_path = os.path.join(web_folder, file_name)
        table = pq.read_table(file_path)
        df = table.to_pandas()
        # 将 DataFrame 添加到列表中
        dfs.append(df)
        if only_first:
            break

# 合并所有 DataFrame
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 检查合并后的数据集是否有足够的行数
    total_rows = len(combined_df)
    if samples_num is not None:
        if total_rows >= samples_num:
            # 从合并后的 DataFrame 中随机选择 samples_num 条记录
            sampled_df = combined_df.sample(n=samples_num, random_state=42)
        else:
            print(f"合并后的数据集只有 {total_rows} 条记录，少于 {samples_num} 条，将使用所有可用数据。")
            sampled_df = combined_df
    else:
        sampled_df = combined_df

    # 将筛选后的 DataFrame 转换为 JSON 格式
    json_data = sampled_df.to_json(orient='records', lines=True)

    # 将 JSON 数据写入文件
    output_json_path = os.path.join(web_folder, 'combined_data.json')
    with open(output_json_path, 'w') as f:
        f.write(json_data)
    print(f"JSON 文件已保存到 {output_json_path}")
else:
    print("目录中未找到 parquet 文件。")