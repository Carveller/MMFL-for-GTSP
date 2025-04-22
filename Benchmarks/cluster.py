import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
data = pd.read_csv('Benchmarks/sampled_points.csv')

# 提取坐标
X = data[['col', 'row']].values

# 使用K-means聚类，将点分为10类
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# 获取每个簇的中心点
cluster_centers = kmeans.cluster_centers_

# 设置仓库点坐标为(40,50)
depot_x, depot_y = 50, 40

# 创建一个大图
plt.figure(figsize=(12, 10))

# 绘制散点图，每个聚类使用不同的颜色
for i in range(n_clusters):
    cluster_points = data[data['cluster'] == i]
    plt.scatter(
        cluster_points['col'], 
        cluster_points['row'],
        label=f'聚类 {i+1}',
        s=60,
        alpha=0.7
    )

# 绘制簇的中心点
plt.scatter(
    cluster_centers[:, 0], 
    cluster_centers[:, 1], 
    s=200, 
    marker='X', 
    color='black', 
    label='聚类中心'
)

# 添加仓库点
plt.scatter(
    depot_x, 
    depot_y, 
    s=300, 
    marker='*', 
    color='red', 
    label='仓库点(50,40)'
)

# 添加标题和轴标签
plt.title('K均值聚类结果 (10类)', fontsize=16)
plt.xlabel('列 (col)', fontsize=14)
plt.ylabel('行 (row)', fontsize=14)

# 调整坐标轴方向（使y轴从上到下增加，符合图像坐标系）
# plt.gca().invert_yaxis()

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 添加图例
plt.legend(loc='upper right')

# 调整图的布局
plt.tight_layout()

# 保存图像
plt.savefig('clustered_points.png', dpi=300)

# 显示图像
plt.show()

# 打印每个聚类的点数
cluster_counts = data['cluster'].value_counts().sort_index()
print("\n各聚类的点数:")
for cluster_id, count in cluster_counts.items():
    print(f"聚类 {cluster_id+1}: {count}个点")

# 将分类结果保存到CSV文件
data_with_clusters = data.copy()
data_with_clusters['cluster'] = data['cluster']
data_with_clusters.to_csv('clustered_sampled_points.csv', index=False)
print(f"\n分类结果已保存到 'clustered_sampled_points.csv'")

# 将结果保存为pt格式
# 准备保存数据
coords = X  # 坐标数据，格式为 [n_points, 2]
cluster_idx = data['cluster'].values  # 聚类标签，格式为 [n_points]

# 转换为PyTorch张量并调整维度为 [1, n_points, 2] 和 [1, n_points]
node_xy = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
cluster_idx = torch.tensor(cluster_idx, dtype=torch.long).unsqueeze(0)

# 添加仓库点坐标
depot_coords = torch.tensor([[depot_x, depot_y]], dtype=torch.float32).unsqueeze(0)
node_xy_with_depot = torch.cat([depot_coords, node_xy], dim=1)

# depot的聚类标签设为-1（特殊标记）
depot_cluster = torch.tensor([[-1]], dtype=torch.long)
cluster_idx_with_depot = torch.cat([depot_cluster, cluster_idx], dim=1)

# 保存为pt文件
output_file = 'clustered_points.pt'
torch.save({
    'node_xy': node_xy_with_depot,  # 包括仓库点的坐标，形状[1, n_points+1, 2]
    'cluster_idx': cluster_idx_with_depot  # 包括仓库点的聚类标签，形状[1, n_points+1]
}, output_file)

print(f'\n聚类结果已保存为 "{output_file}"')
print(f'- node_xy 张量形状: {node_xy_with_depot.shape}')
print(f'- cluster_idx 张量形状: {cluster_idx_with_depot.shape}')
print(f'- 注意：第一个点是仓库点(50,40)，聚类标签为-1')