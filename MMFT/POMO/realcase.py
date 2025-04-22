import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # for problem_def
sys.path.insert(0, os.path.join(current_dir, ".."))  # for utils

# 导入必要的模块
from GTSPEnv import TSPEnv as Env, Reset_State
from GTSPModel import GTSPGIMFModel as Model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ClusteredPointsSolver:
    def __init__(self, device='cuda'):
        # 设置是否使用CUDA
        self.use_cuda = torch.cuda.is_available() and device == 'cuda'
        if self.use_cuda:
            torch.cuda.set_device(0)
            self.device = torch.device('cuda', 0)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # 环境参数
        self.env_params = {
            'problem_size': 50,  # 点数量
            'pomo_size': 1000000,    # POMO尺寸
        }
        
        # 模型参数
        self.model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128**(1/2),
            'encoder_layer_num': 3,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'softmax',
            # 新增参数
            'fusion_layer_num': 3,
            'bottleneck_size': 10,
            'patch_size': 16,
            'in_channels': 1,
            # 问题规模自适应分辨率(PSAR)策略参数
            'use_psar': True,
        }
        
        # 初始化环境和模型
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        # 如果使用CUDA，将模型移至GPU
        if self.use_cuda:
            self.model.cuda()
            
        # 加载预训练模型
        self.load_pretrained_model()
        
        # 设置模型为评估模式
        self.model.eval()
    
    def load_pretrained_model(self):
        # 加载预训练模型，替换为您的模型路径
        checkpoint_epoch = 200
        checkpoint_fullname = os.path.join(r"D:\OneDrive\0001博士\0001论文\0007GTSP\GTSP_ViT_CrossAtt_1\MMFT\POMO\result\100_2", 
                                           f'checkpoint-{checkpoint_epoch}.pt')
        
        print(f"加载模型: {checkpoint_fullname}")
        if os.path.exists(checkpoint_fullname):
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载成功!")
        else:
            print(f"警告: 模型文件不存在 - {checkpoint_fullname}")
            print("将使用未训练的模型进行推理，结果可能不准确")
    
    def load_clustered_points(self, filepath):
        """加载聚类后的点数据"""
        print(f"从{filepath}加载数据")
        data = torch.load(filepath, map_location=self.device)
        
        # 直接使用文件中的数据，不再添加depot节点
        node_xy = data['node_xy']
        cluster_idx = data['cluster_idx']
        
        # 获取问题规模
        problem_size = node_xy.shape[1] - 1  # 减1因为包含了depot
        cluster_count = cluster_idx.max().item() + 1  # +1因为聚类是从0开始的
        
        print(f"加载了{problem_size}个点，分为{cluster_count}个聚类")
        
        # 返回原始的节点坐标（无depot）供可视化使用
        original_xy = node_xy[:, 1:, :]  # 排除第一个点（depot）
        
        return node_xy, cluster_idx, problem_size, original_xy
    
    def normalize_xy(self, node_xy):
        """归一化坐标到[0,1]范围"""
        min_xy = node_xy.min(dim=1, keepdim=True)[0]
        max_xy = node_xy.max(dim=1, keepdim=True)[0]
        
        # 避免除零错误
        denominator = max_xy - min_xy
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        
        # 归一化
        normalized_xy = (node_xy - min_xy) / denominator
        
        return normalized_xy
    
    def solve(self, node_xy, cluster_idx, original_xy):
        """解决GTSP问题并返回结果"""
        # 归一化坐标
        normalized_xy = self.normalize_xy(node_xy)
        
        # 准备环境
        self.env.node_xy = normalized_xy
        self.env.cluster_idx = cluster_idx
        self.env.batch_size = 1
        
        # 设置BATCH_IDX和POMO_IDX
        pomo_size = self.env_params['pomo_size']
        self.env.BATCH_IDX = torch.arange(1)[:, None].expand(1, pomo_size).to(self.device)
        self.env.POMO_IDX = torch.arange(pomo_size)[None, :].expand(1, pomo_size).to(self.device)
        
        # 重置环境
        reset_state, _, _ = self.env.reset()
        
        # 创建一个新的reset_state对象
        custom_reset_state = Reset_State()
        custom_reset_state.node_xy = normalized_xy
        custom_reset_state.cluster_idx = cluster_idx
        
        # 开始计时
        if self.use_cuda:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            start_time = time.time()
        
        # 准备模型
        self.model.pre_forward(custom_reset_state)
        
        # 运行规划
        state, reward, done = self.env.pre_step()
        
        if state.BATCH_IDX is None:
            state.BATCH_IDX = self.env.BATCH_IDX
            state.POMO_IDX = self.env.POMO_IDX
        
        while not done:
            selected, _ = self.model(state, normalized_xy)
            state, reward, done = self.env.step(selected)
        
        # 结束计时
        if self.use_cuda:
            end_time.record()
            torch.cuda.synchronize()
            solve_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        else:
            solve_time = time.time() - start_time
        
        # 获取结果
        rewards = reward.reshape(1, pomo_size)  # [batch, pomo]
        best_reward_val, best_idx = rewards.max(dim=1)  # [batch], [batch]
        
        # 提取最佳路径
        best_tour = self.env.selected_node_list[0, best_idx[0]].cpu().numpy()
        
        # 计算路径长度（使用原始坐标）
        distance = self.calculate_path_length(original_xy[0], best_tour)
        
        # 生成路径坐标
        path_coords = self.get_path_coordinates(original_xy[0], best_tour)
        
        # 返回结果
        return {
            'tour': best_tour,
            'distance': distance,
            'solve_time': solve_time,
            'negative_reward': -best_reward_val.item(),  # 负的奖励值应该与距离相等
            'path_coords': path_coords  # 添加路径坐标
        }

    # 在get_path_coordinates函数中修改
    def get_path_coordinates(self, coordinates, tour):
        """获取路径中每个点的坐标，其中row是横坐标，col是纵坐标"""
        path_coords = []
        
        # 仓库位于(40,50)，其中40是row(横坐标)，50是col(纵坐标)
        depot_row, depot_col = 50, 40
        
        for idx in tour:
            if idx == 0:  # depot
                path_coords.append({
                    'index': 0,
                    'row': depot_row,
                    'col': depot_col,
                    'type': 'depot'
                })
            else:
                # 调整索引（因为coordinates中没有depot）
                adj_idx = idx - 1
                coords = coordinates[adj_idx].cpu().numpy()
                path_coords.append({
                    'index': int(idx),
                    'row': float(coords[1]),  # row是第二个元素
                    'col': float(coords[0]),  # col是第一个元素
                    'type': 'point'
                })
        
        return path_coords
    
    def calculate_path_length(self, coordinates, tour):
        """计算路径长度，包括仓库节点(40,50)"""
        # 创建路径点的坐标列表
        path_coords = []
        
        # 仓库位于(40,50)
        depot_coords = np.array([50, 40])
        
        for idx in tour:
            if idx == 0:  # depot
                path_coords.append(depot_coords)
            else:
                # 调整索引（因为coordinates中没有depot）
                adj_idx = idx - 1
                path_coords.append(coordinates[adj_idx].cpu().numpy())
        
        # 转换为numpy数组
        path_coords = np.array(path_coords)
        
        # 计算相邻点之间的距离
        distances = np.sqrt(np.sum(np.diff(path_coords, axis=0)**2, axis=1))
        
        # 添加从最后一个点回到第一个点的距离（闭环）
        if len(path_coords) > 1:
            last_to_first = np.sqrt(np.sum((path_coords[-1] - path_coords[0])**2))
            total_distance = np.sum(distances) + last_to_first
        else:
            total_distance = 0
            
        return total_distance
    
    def visualize_solution(self, original_xy, tour, clusters, output_path='gtsp_solution.png'):
        """可视化解决方案，并确保包含仓库节点(40,50)，其中40是row，50是col"""
        # 获取原始坐标
        coords = original_xy[0].cpu().numpy()
        
        # 获取聚类信息
        cluster_labels = clusters[0, 1:].cpu().numpy()  # 排除depot的聚类标签
        
        # 处理tour
        tour_np = tour.copy()
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 绘制聚类点
        for cluster_id in range(int(cluster_labels.max()) + 1):
            cluster_points = np.where(cluster_labels == cluster_id)[0]
            plt.scatter(
                coords[cluster_points, 1],  # row为横坐标
                coords[cluster_points, 0],  # col为纵坐标
                label=f'聚类 {cluster_id+1}',
                s=50,
                alpha=0.7
            )
        
        # 添加仓库节点(40,50)，其中40是row，50是col
        depot_row, depot_col = 50, 40
        plt.scatter(depot_row, depot_col, color='red', marker='*', s=200, label='仓库(50,40)')
        
        # 画路径（按顺序连接，包括仓库）
        for i in range(len(tour_np)):
            start_idx = tour_np[i]
            end_idx = tour_np[(i + 1) % len(tour_np)]
            
            # 获取起点坐标
            if start_idx == 0:  # depot
                start_row, start_col = depot_row, depot_col
            else:
                # 调整索引（因为coords中没有depot）
                adj_start_idx = start_idx - 1
                start_row, start_col = coords[adj_start_idx, 1], coords[adj_start_idx, 0]  # 交换row和col
            
            # 获取终点坐标
            if end_idx == 0:  # depot
                end_row, end_col = depot_row, depot_col
            else:
                # 调整索引（因为coords中没有depot）
                adj_end_idx = end_idx - 1
                end_row, end_col = coords[adj_end_idx, 1], coords[adj_end_idx, 0]  # 交换row和col
            
            # 绘制线段
            plt.plot(
                [start_row, end_row],
                [start_col, end_col],
                'k-',
                alpha=0.5
            )
        
        # 标记路径点和顺序
        for i, idx in enumerate(tour_np):
            if idx == 0:  # depot
                plt.annotate(
                    f'{i+1}',
                    (depot_row, depot_col),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    weight='bold'
                )
            else:
                # 调整索引（因为coords中没有depot）
                adj_idx = idx - 1
                plt.annotate(
                    f'{i+1}',
                    (coords[adj_idx, 1], coords[adj_idx, 0]),  # 交换row和col
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='blue'
                )
        
        # 添加标题和标签
        plt.title(f'GTSP解决方案（距离: {result["distance"]:.2f}）', fontsize=16)
        plt.xlabel('行 (row)', fontsize=14)
        plt.ylabel('列 (col)', fontsize=14)
        
        # 反转y轴（使其符合图像坐标系）
        plt.gca().invert_yaxis()
        
        # 添加图例
        plt.legend(loc='upper right')
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=300)
        print(f'解决方案可视化保存至 "{output_path}"')
        
        # 显示图像
        plt.show()

import csv

# 在主程序的末尾添加以下代码来将路径坐标写入CSV文件
def save_path_to_csv(path_coords, filename='gtsp_path.csv'):
    """将路径坐标保存到CSV文件"""
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        fieldnames = ['order', 'index', 'row', 'col', 'type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据
        for i, point in enumerate(path_coords):
            writer.writerow({
                'order': i + 1,
                'index': point['index'],
                'row': point['row'],
                'col': point['col'],
                'type': point['type']
            })
    
    print(f"\n路径坐标已保存到 {filename}")

# 在主程序中调用这个函数
if __name__ == "__main__":
    # 路径到pt文件
    pt_file = r"D:\OneDrive\0001博士\0001论文\0007GTSP\GTSP_ViT_CrossAtt_1\Benchmarks\clustered_points.pt"
    
    # 创建求解器
    solver = ClusteredPointsSolver(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    node_xy, cluster_idx, problem_size, original_xy = solver.load_clustered_points(pt_file)
    
    # 求解GTSP问题
    result = solver.solve(node_xy, cluster_idx, original_xy)
    
    # 打印结果
    print(f"\n求解完成!")
    print(f"路径长度: {result['distance']:.4f}")
    print(f"求解时间: {result['solve_time']:.4f}秒")
    
    # 打印路径索引和坐标
    print("\n路径点索引和坐标:")
    for i, point in enumerate(result['path_coords']):
        if point['type'] == 'depot':
            print(f"{i+1}. 仓库 (索引: {point['index']}, 横坐标(row): {point['row']}, 纵坐标(col): {point['col']})")
        else:
            print(f"{i+1}. 点 (索引: {point['index']}, 横坐标(row): {point['row']}, 纵坐标(col): {point['col']})")
    
    # 将路径保存到CSV文件
    save_path_to_csv(result['path_coords'], 'gtsp_path.csv')
    
    # 可视化解决方案
    solver.visualize_solution(original_xy, result['tour'], cluster_idx)