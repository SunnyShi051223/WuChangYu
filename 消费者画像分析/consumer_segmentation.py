import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_persona_clustering():
    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'data', '汇总版_已清洗数值化.xlsx')
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 {file_path}")
        return

    df = pd.read_excel(file_path)

    # 1. 变量选取与聚合得分计算
    # 产地敏感度: Q10, Q11, Q12
    # 文化涉入度: Q13, Q14, Q15
    # 品牌知识/声望依赖: Q16, Q17, Q18
    # 价格敏感度: Q19, Q20, Q21
    
    cols = df.columns
    
    # 辅助函数：根据题号找列名
    def get_col_by_num(num):
        return [c for c in cols if str(c).startswith(f"{num}.")][0]

    # 计算各维度平均分
    df['产地敏感度'] = df[[get_col_by_num(10), get_col_by_num(11), get_col_by_num(12)]].mean(axis=1)
    df['文化涉入度'] = df[[get_col_by_num(13), get_col_by_num(14), get_col_by_num(15)]].mean(axis=1)
    df['品牌依赖度'] = df[[get_col_by_num(16), get_col_by_num(17), get_col_by_num(18)]].mean(axis=1)
    df['价格敏感度'] = df[[get_col_by_num(19), get_col_by_num(20), get_col_by_num(21)]].mean(axis=1)

    # 聚类所需特征
    features = ['产地敏感度', '文化涉入度', '品牌依赖度', '价格敏感度']
    X = df[features]

    # 2. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 执行 K-Means 聚类 (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 4. 分析聚类中心 (Profile)
    cluster_centers = df.groupby('Cluster')[features].mean()
    print("\n--- 聚类中心特征值 ---")
    print(cluster_centers)
    
    # 获取各簇人数占比
    counts = df['Cluster'].value_counts(normalize=True).sort_index()
    print("\n--- 各类别占比 ---")
    print(counts)

    # 5. 可视化：雷达图
    labels = np.array(features)
    num_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合圆圈

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    cluster_names = ['跟风大众型', '文化溯源型', '价格导向型'] # 预定义顺序需结合均值判断，这里先做图

    for i in range(3):
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
    
    # 6. 保存带有分类的数据供后续分析（如画像描述）
    df.to_excel(os.path.join(script_dir, 'clustered_customers.xlsx'), index=False)
    print(f"\n✅ 聚类分析完成，已保存 clustered_customers.xlsx 和图表。")

if __name__ == "__main__":
    run_persona_clustering()
