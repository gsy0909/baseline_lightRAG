import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. 读取并解析数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('[source'): continue # 跳过干扰字符
            try:
                # 尝试处理包含多余字符的行
                clean_line = line.split('} {')[0] if '}' in line else line
                item = json.loads(clean_line)
                if "query_id" in item:
                    data.append({
                        "query_id": item["query_id"],
                        "recall": item["recall"],
                        "relevant_total": item["relevant_total"],
                        "category": item["query_id"][0] # 提取首字母 g, o, s
                    })
            except:
                continue
    return pd.DataFrame(data)

# 2. 执行分析
def run_analysis():
    # 假设你的文件名叫 retrieval_results.jsonl
    df = load_jsonl('retrieval_results_vague.jsonl')
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial'] # 如果报错可改为系统可用字体
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # --- 图 1: Recall 分布直方图 ---
    sns.histplot(df['recall'], bins=10, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Global Recall Distribution (Total Queries: 100)')
    axes[0].set_xlabel('Recall Score')

    # --- 图 2: 不同数据集的平均 Recall ---
    avg_recall = df.groupby('category')['recall'].mean().reset_index()
    sns.barplot(x='category', y='recall', data=avg_recall, ax=axes[1], palette='magma')
    axes[1].set_title('Average Recall by Dataset Category (g, o, s)')
    for i, v in enumerate(avg_recall['recall']):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center')

    # --- 图 3: 相关文档数对效果的影响 ---
    # 过滤掉样本极少的 relevant_total 以免图表混乱
    sns.lineplot(x='relevant_total', y='recall', hue='category', marker='o', data=df, ax=axes[2])
    axes[2].set_title('Recall vs Relevant Document Count')
    
    plt.tight_layout()
    plt.savefig('recall_analysis_report.png') # 自动保存为图片
    plt.show()

if __name__ == "__main__":
    run_analysis()