import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_cognitive_analysis():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    # 数据路径 (相对于脚本所在目录)
    file_path = os.path.join(script_dir, '..', 'data', '汇总版_已清洗数值化.xlsx')
    
    if not os.path.exists(file_path):
        # 尝试不同路径
        file_path = os.path.join('data', '汇总版_已清洗数值化.xlsx')
        if not os.path.exists(file_path):
            print("错误: 找不到数据文件")
            return

    df = pd.read_excel(file_path)
    
    # --- 2.2.1 地理认知错位 (Heatmap) ---
    # 5. 居住地: 1=武汉, 2=鄂州, 3=湖北省内其他城市
    # 8. 核心原产地认知: 1=湖北武汉, 2=湖北鄂州, 3=湖北黄石, 4=其他, 5=不清楚
    
    loc_col = [c for c in df.columns if '5. 您目前居住' in str(c)][0]
    origin_col = [c for c in df.columns if '8. 您认为武昌鱼的核心' in str(c)][0]
    
    # 映射字典
    loc_map = {1: '武汉居民', 2: '鄂州居民', 3: '省内其他'}
    origin_map = {1: '认为产地:武汉', 2: '认为产地:鄂州', 3: '认为产地:黄石', 4: '其他', 5: '不清楚'}
    
    # 应用映射进行重命名
    df_temp = df.copy()
    df_temp[loc_col] = df_temp[loc_col].map(loc_map)
    df_temp[origin_col] = df_temp[origin_col].map(origin_map)
    
    # 创建交叉表
    ct = pd.crosstab(df_temp[loc_col], df_temp[origin_col], normalize='index') * 100
    
    # 按照特定顺序排序（如果存在这些列）
    row_order = [v for v in loc_map.values() if v in ct.index]
    col_order = [v for v in origin_map.values() if v in ct.columns]
    ct = ct.reindex(index=row_order, columns=col_order)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '百分比 (%)'})
    plt.title('受访者居住地 vs 武昌鱼原产地认知错位热力图')
    plt.ylabel('受访者居住地')
    plt.xlabel('认知中的核心原产地')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'geographical_mismatch_heatmap.png'))
    plt.close()
    print("✅ 地理认知错位热力图已生成。")

    # --- 2.2.2 品牌联想断层 (Word Clouds & 断层率) ---
    font_path = r'C:\Windows\Fonts\simhei.ttf'
    
    # Q9 多选题各选项
    q9_cols = {
        '毛主席/诗词': [c for c in df.columns if '9' in str(c) and '毛主席' in str(c)][0],
        '武汉/黄鹤楼': [c for c in df.columns if '9' in str(c) and '武汉/黄鹤楼' in str(c)][0],
        '鄂州/梁子湖': [c for c in df.columns if '9' in str(c) and '鄂州' in str(c) and '梁子湖' in str(c)][0],
        '三国/孙权': [c for c in df.columns if '9' in str(c) and '三国' in str(c)][0],
        '烹饪方式': [c for c in df.columns if '9' in str(c) and '烹饪' in str(c)][0],
    }
    
    # 统计各项频率
    freqs = {name: df[col].sum() for name, col in q9_cols.items()}
    
    # 定义“大众认知”和“原产地实情”词库
    # 大众认知：毛主席诗词、武汉/黄鹤楼、烹饪方式
    # 原产地实情：鄂州/梁子湖、三国关系/孙权
    popular_tags = {'毛主席/诗词': freqs['毛主席/诗词'], '武汉/黄鹤楼': freqs['武汉/黄鹤楼'], '烹饪方式': freqs['烹饪方式']}
    origin_tags = {'鄂州/梁子湖': freqs['鄂州/梁子湖'], '三国/孙权': freqs['三国/孙权']}
    
    # 生成词云 (大众认知)
    wc_pop = WordCloud(font_path=font_path, width=600, height=400, background_color='white').generate_from_frequencies(popular_tags)
    plt.figure(figsize=(8, 5))
    plt.imshow(wc_pop, interpolation='bilinear')
    plt.axis('off')
    plt.title('大众核心联想词云 (偏向符号/行政大市)')
    plt.savefig(os.path.join(output_dir, 'popular_association_wordcloud.png'))
    plt.close()

    # 生成词云 (产地实情)
    wc_ori = WordCloud(font_path=font_path, width=600, height=400, background_color='white').generate_from_frequencies(origin_tags)
    plt.figure(figsize=(8, 5))
    plt.imshow(wc_ori, interpolation='bilinear')
    plt.axis('off')
    plt.title('原产地核心联想词云 (地理/历史实情)')
    plt.savefig(os.path.join(output_dir, 'origin_association_wordcloud.png'))
    plt.close()
    
    # 计算认知断层率
    pop_mean = sum(popular_tags.values()) / len(popular_tags)
    ori_mean = sum(origin_tags.values()) / len(origin_tags)
    gap_rate = (pop_mean - ori_mean) / pop_mean if pop_mean > 0 else 0
    
    print(f"✅ 品牌联想词云已生成。")
    print(f"📊 大众认知平均提及数: {pop_mean:.1f}")
    print(f"📊 原产地实情平均提及数: {ori_mean:.1f}")
    print(f"📉 文化认知断层率: {gap_rate*100:.1f}%")

    with open(os.path.join(output_dir, 'analysis_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f"大众认知平均提及数: {pop_mean:.2f}\n")
        f.write(f"原产地实情平均提及数: {ori_mean:.2f}\n")
        f.write(f"文化认知断层率: {gap_rate*100:.2f}%\n")

if __name__ == "__main__":
    run_cognitive_analysis()
