##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
# import torch
# torch.backends.cudnn.deterministic = True

from GTSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 200,
    'pomo_size': int(20/4),
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 3,
    'qkv_dim': 16,
    'head_num': 8,
    'max_grad_norm': 1.0,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    # 新增参数
    'fusion_layer_num': 3,
    'bottleneck_size': 10,
    'patch_size': 16,
    'in_channels': 1,
    # 问题规模自适应分辨率(PSAR)策略参数
    'use_psar': True,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6,
    },
    'scheduler': {
        'T_max': 200,
        'eta_min': 1e-6  # 最小学习率
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 256,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 510,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
