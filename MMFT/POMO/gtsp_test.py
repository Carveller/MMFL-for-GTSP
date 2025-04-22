#!/usr/bin/env python
##########################################################################################
# GTSP Testing Program (Simplified & Fixed)
# This script loads PT format GTSP instances and evaluates them using the existing model

import os
import sys
import time
import numpy as np
import torch
import argparse
import json
from datetime import datetime

# torch.backends.cudnn.deterministic = True

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # for problem_def
sys.path.insert(0, os.path.join(current_dir, ".."))  # for utils

# Import necessary modules
from GTSPEnv import TSPEnv as Env, Reset_State
from GTSPModel import GTSPGIMFModel as Model


class GTSPTester:
    def __init__(self, env_params, model_params, tester_params):
        # Save parameters
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        
        # Initialize environment and model
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # CUDA setup
        self.use_cuda = tester_params['use_cuda']
        if self.use_cuda:
            cuda_device_num = tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        

        # Load pre-trained model
        if tester_params['model_load']['enable']:
            checkpoint_path = tester_params['model_load']['path']
            checkpoint_epoch = tester_params['model_load']['epoch']
            checkpoint_fullname = os.path.join(checkpoint_path, f'checkpoint-{checkpoint_epoch}.pt')
            
            print(f"Loading model from {checkpoint_fullname}")
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!")
        
        self.model.eval()  # Set model to evaluation mode

    def load_instances_from_pt(self, filepath):
        """
        Load GTSP instances from a PT file
        """
        print(f"Loading instances from {filepath}")
        data = torch.load(filepath, map_location=self.device)
        
        node_xy = data['node_xy']
        cluster_idx = data['cluster_idx']
        
        problem_size = node_xy.shape[1] - 1  # Excluding depot
        cluster_count = cluster_idx.max().item() + 1  # +1 because clusters are 0-indexed
        
        return node_xy.float(), cluster_idx.float(), problem_size, cluster_count
    
    def solve_batch(self, node_xy, cluster_idx, batch_idx):
        """
        Solve a batch of GTSP instances
        """
        batch_size = self.tester_params.get('test_batch_size', 10)
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, node_xy.shape[0])
        
        actual_batch_size = end_idx - start_idx
        print(f"Solving batch {batch_idx+1}, instances {start_idx+1}-{end_idx} (batch size: {actual_batch_size})")
        
        # Get batch data
        batch_node_xy = node_xy[start_idx:end_idx].to(self.device)
        batch_cluster_idx = cluster_idx[start_idx:end_idx].to(self.device)
        
        results = []
        solve_times = []
        
        # Solve each instance in the batch
        for i in range(actual_batch_size):
            instance_node_xy = batch_node_xy[i:i+1]  # Keep batch dimension
            instance_cluster_idx = batch_cluster_idx[i:i+1]  # Keep batch dimension
            
            # Prepare environment
            self.env.node_xy = instance_node_xy
            self.env.cluster_idx = instance_cluster_idx
            self.env.batch_size = 1
            
            # Set up BATCH_IDX and POMO_IDX for the environment
            pomo_size = self.env_params['pomo_size']
            self.env.BATCH_IDX = torch.arange(1)[:, None].expand(1, pomo_size)
            self.env.POMO_IDX = torch.arange(pomo_size)[None, :].expand(1, pomo_size)
            
            # Reset environment
            reset_state, _, _ = self.env.reset()
            
            # 修复：创建一个新的reset_state对象，确保所有必要的属性都存在
            # 根据我们遇到的错误，需要确保node_xy和cluster_idx属性有效
            custom_reset_state = Reset_State()
            custom_reset_state.node_xy = instance_node_xy  # 确保node_xy不是None
            custom_reset_state.cluster_idx = instance_cluster_idx  # 确保cluster_idx不是None
            
            # Start timer
            start_time = torch.cuda.Event(enable_timing=True) if self.use_cuda else time.time()
            end_time = torch.cuda.Event(enable_timing=True) if self.use_cuda else None
            
            if self.use_cuda:
                start_time.record()
            
            # Prepare model
            self.model.pre_forward(custom_reset_state)
            
            # Run planning
            state, reward, done = self.env.pre_step()
            
            if state.BATCH_IDX is None:
                state.BATCH_IDX = self.env.BATCH_IDX
                state.POMO_IDX = self.env.POMO_IDX

            while not done:
                selected, _ = self.model(state, instance_node_xy)
                state, reward, done = self.env.step(selected)
            
            # End timer
            if self.use_cuda:
                end_time.record()
                torch.cuda.synchronize()
                solve_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                solve_time = time.time() - start_time
            
            # Get results
            rewards = reward.reshape(1, pomo_size)  # [batch, pomo]
            best_reward_val, best_idx = rewards.max(dim=1)  # [batch], [batch]
            
            # Extract best tour
            best_tour = self.env.selected_node_list[0, best_idx[0]].cpu().numpy()
            
            # Store results
            results.append({
                'instance_idx': start_idx + i,
                'travel_distance': -best_reward_val.item(),  # Negate to get actual distance
                'solve_time': solve_time
            })
            solve_times.append(solve_time)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == actual_batch_size - 1:
                print(f"  Processed {i+1}/{actual_batch_size} instances, average time: {np.mean(solve_times):.3f}s")
        
        return results
    
    def run_test(self, instance_path, output_path):
        """
        Run test on GTSP instances
        """
        # Load instances
        node_xy, cluster_idx, problem_size, cluster_count = self.load_instances_from_pt(instance_path)
        print(f"Loaded {node_xy.shape[0]} instances with problem size {problem_size} and {cluster_count} clusters")
        
        # Determine number of batches
        batch_size = self.tester_params.get('test_batch_size', 10)
        num_instances = node_xy.shape[0]
        num_batches = (num_instances + batch_size - 1) // batch_size  # Ceiling division
        
        # Store all results
        all_results = []
        
        # Process in batches
        for batch_idx in range(num_batches):
            batch_results = self.solve_batch(node_xy, cluster_idx, batch_idx)
            all_results.extend(batch_results)
        
        # Calculate statistics
        travel_distances = [r['travel_distance'] for r in all_results]
        solve_times = [r['solve_time'] for r in all_results]
        
        stats = {
            'problem_size': problem_size,
            'cluster_count': cluster_count,
            'num_instances': num_instances,
            'avg_travel_distance': float(np.mean(travel_distances)),
            'std_travel_distance': float(np.std(travel_distances)),
            'min_travel_distance': float(np.min(travel_distances)),
            'max_travel_distance': float(np.max(travel_distances)),
            'avg_solve_time': float(np.mean(solve_times)),
            'std_solve_time': float(np.std(solve_times)),
            'min_solve_time': float(np.min(solve_times)),
            'max_solve_time': float(np.max(solve_times)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.tester_params['model_load']['path'],
            'model_epoch': self.tester_params['model_load']['epoch'],
            'device': 'cuda' if self.use_cuda else 'cpu'
        }
        
        # Save detailed results
        results_data = {
            'stats': stats,
            'details': all_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nTesting completed. Results saved to {output_path}")
        print(f"\nSummary Statistics:")
        print(f"  Average Travel Distance: {stats['avg_travel_distance']:.4f} ± {stats['std_travel_distance']:.4f}")
        print(f"  Average Solve Time: {stats['avg_solve_time']:.4f}s ± {stats['std_solve_time']:.4f}s")
        print(f"  Min/Max Travel Distance: {stats['min_travel_distance']:.4f} / {stats['max_travel_distance']:.4f}")
        print(f"  Min/Max Solve Time: {stats['min_solve_time']:.4f}s / {stats['max_solve_time']:.4f}s")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='GTSP Instance Tester')
    parser.add_argument('--instance_path', type=str,
                        default=r"D:\OneDrive\0001博士\0001论文\0007GTSP\GTSP_ViT_CrossAtt_1\Benchmarks\converted\20kroA100.pt",
                        help='Path to PT instance file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output JSON results file')
    parser.add_argument('--pomo_size', type=int, default=1000, help='POMO size')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
    parser.add_argument('--model_path', type=str,
                        default=r"D:\OneDrive\0001博士\0001论文\0007GTSP\GTSP_ViT_CrossAtt_1\MMFT\POMO\result\200",
                        help='Path to pre-trained model directory')
    parser.add_argument('--model_epoch', type=int, default=200, help='Epoch of pre-trained model')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for testing')
    
    args = parser.parse_args()
    
    # If output path not specified, create one based on the instance path
    if args.output_path is None:
        instance_name = os.path.splitext(os.path.basename(args.instance_path))[0]
        args.output_path = f"results_{instance_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Auto-detect problem size from the PT file
    pt_data = torch.load(args.instance_path, map_location='cpu')
    problem_size = pt_data['node_xy'].shape[1] - 1  # Exclude depot
    
    # Setup parameters
    env_params = {
        'problem_size': problem_size,
        'pomo_size': args.pomo_size,
    }

    model_params = {
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

    tester_params = {
        'use_cuda': not args.no_cuda,
        'cuda_device_num': args.cuda_device,
        'model_load': {
            'enable': True,
            'path': args.model_path,
            'epoch': args.model_epoch,
        },
        'test_batch_size': args.batch_size
    }
    
    # Initialize tester
    tester = GTSPTester(env_params, model_params, tester_params)
    
    # Run test
    tester.run_test(args.instance_path, args.output_path)


if __name__ == "__main__":
    main()