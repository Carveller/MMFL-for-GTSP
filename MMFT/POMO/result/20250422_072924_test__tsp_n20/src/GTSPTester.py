
import torch

import os
from logging import getLogger

from GTSPEnv import TSPEnv as Env
from GTSPModel import GTSPGIMFModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.model = self.model.cuda()
        else:
            device = torch.device('cpu')
            # torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device



        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score, max_pomo_loc_list, max_aug_loc_list, node_xy, radius = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        return max_pomo_loc_list, max_aug_loc_list, node_xy, radius

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        node_xy = self.env.node_xy
        while not done:
            selected, prob = self.model(state, node_xy)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
        loc_list = self.env.selected_node_list.reshape(aug_factor, batch_size, self.env.pomo_size, -1, 1)

        max_pomo_reward, max_pomo_idx = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_pomo_idx_expand = max_pomo_idx.view(1, batch_size, 1, 1, 1).expand(aug_factor, batch_size, 1,
                                                                               loc_list.size(-2), 1)
        max_pomo_loc_list = loc_list.gather(2, max_pomo_idx_expand)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, max_aug_idx = max_pomo_reward.max(dim=0)  # get best results from augmentation
        max_aug_idx_expanded = max_aug_idx[None, :, None, None, None].expand(1, batch_size, 1, loc_list.size(-2), 1)
        # shape: (batch,)
        max_aug_loc_list = max_pomo_loc_list.gather(0, max_aug_idx_expanded)

        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item(), max_pomo_loc_list.cpu().numpy(), \
               max_aug_loc_list.cpu().numpy(), self.env.node_xy.cpu().numpy(), None
