
from dataclasses import dataclass
import torch

from GTSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem+1, 2)
    cluster_idx: torch.Tensor = None
    # shape: (batch, problem+1)



@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.node_xy = None
        self.cluster_idx = None
        # shape: (batch, node, node)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        node_xy, cluster_idx = get_random_problems(batch_size, self.problem_size)
        self.node_xy = node_xy
        self.cluster_idx = cluster_idx
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.node_xy = augment_xy_data_by_8_fold(self.node_xy)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.reset_state.node_xy = node_xy
        self.reset_state.cluster_idx = cluster_idx

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.node_xy.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.node_xy.device)

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.node_xy.device, requires_grad=False)
        # shape: (batch, pomo, 0~problem)

        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=self.node_xy.device, requires_grad=False)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.node_xy.device, requires_grad=False)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        
        # 重要：确保BATCH_IDX和POMO_IDX与环境的当前配置一致
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, test=False):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        node_xy_expand = self.node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 2)
        chose_node = node_xy_expand[self.BATCH_IDX, self.POMO_IDX, self.current_node]
        cluster_idx_expand = self.cluster_idx[:, None, :].expand(self.batch_size, self.pomo_size, self.problem_size+1)
        selected_clu = cluster_idx_expand[self.BATCH_IDX, self.POMO_IDX, self.current_node]

        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        same_clu_idx = (cluster_idx_expand == selected_clu[:, :, None].expand_as(cluster_idx_expand))
        self.ninf_mask[same_clu_idx] = float('-inf')

        # do not mask depot for finished episode.
        self.finished = self.finished + (self.ninf_mask[:, :, 1:] == float('-inf')).all(dim=2)
        self.ninf_mask[:, :, 0][self.finished] = 0
        
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(-1, -1, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

