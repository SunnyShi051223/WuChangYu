import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import silhouette_score

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def determine_optimal_k(X_scaled, script_dir):
    sse = []  # 误差平方和 (用于手肘法)
    silhouette_scores = []  # 轮廓系数
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
    # 绘制手肘法图和轮廓系数图
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('聚类数量 (k)')
    ax1.set_ylabel('误差平方和 (SSE)', color=color)
    ax1.plot(k_range, sse, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('轮廓系数 (Silhouette Score)', color=color)
    ax2.plot(k_range, silhouette_scores, marker='s', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('K-means 最优簇数确定 (手肘法与轮廓系数)')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'optimal_k_evaluation.png'))
    plt.close()
    print(f"✅ 已生成最优簇数评估图 ({os.path.join(script_dir, 'optimal_k_evaluation.png')})")

def run_persona_clustering():
    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'data', '汇总.xlsx')
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 {file_path}")
        return

    df = pd.read_excel(file_path)

    # 1. 变量选取与聚合得分计算
    cols = df.columns
    def get_col_by_num(num):
        return [c for c in cols if str(c).startswith(f"{num}.")][0]

    df['产地敏感度'] = df[[get_col_by_num(10), get_col_by_num(11), get_col_by_num(12)]].mean(axis=1)
    df['文化涉入度'] = df[[get_col_by_num(13), get_col_by_num(14), get_col_by_num(15)]].mean(axis=1)
    df['品牌依赖度'] = df[[get_col_by_num(16), get_col_by_num(17), get_col_by_num(18)]].mean(axis=1)
    df['价格敏感度'] = df[[get_col_by_num(19), get_col_by_num(20), get_col_by_num(21)]].mean(axis=1)

    features = ['产地敏感度', '文化涉入度', '品牌依赖度', '价格敏感度']
    X = df[features]

    # 2. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 之前确定 k=3 是最优或合理的选择
    determine_optimal_k(X_scaled, script_dir)

    # 3. 执行 K-Means 聚类 (k=3)
    k_final = 3
    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 4. 分析聚类中心 (Profile)
    cluster_centers = df.groupby('Cluster')[features].mean()
    print("\n--- 聚类中心特征值 ---")
    print(cluster_centers)
    
    counts = df['Cluster'].value_counts(normalize=True).sort_index()
    print("\n--- 各类别占比 ---")
    print(counts)

    # 5. 可视化：雷达图
    labels = np.array(features)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(k_final):
        values = cluster_centers.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=f'Cluster {i+1}')
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title('消费者特征聚类图谱 (Radar Chart)', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(os.path.join(script_dir, 'consumer_segments_radar.png'), bbox_inches='tight')
    plt.close()
    
    df.to_excel(os.path.join(script_dir, 'clustered_customers.xlsx'), index=False)
    print(f"\n✅ 聚类分析完成，已保存 clustered_customers.xlsx 和图表。")

if __name__ == "__main__":
    run_persona_clustering()
