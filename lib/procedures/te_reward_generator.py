import os
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from procedures import Linear_Region_Collector, get_ntk_n
from models       import get_cell_based_tiny_net, get_search_spaces, nas_super_nets
from pdb import set_trace as bp
from typing import List, Dict, Tuple, Any, Optional

from models.nasbench101 import Network
from models.nasbench101 import ModelSpec

matrix = [[0, 1, 1, 1, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0]]

operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu',
              'maxpool3x3', 'output']

index2ops_101 = {0: 'input',
            1: 'conv1x1-bn-relu',
            2: 'conv3x3-bn-relu',
            3: 'maxpool3x3',
            4: 'output'
            }

# ops_dict = {'input': 0,
#             'conv1x1-bn-relu': 1,
#             'conv3x3-bn-relu': 2,
#             'maxpool3x3': 3,
#             'output': 4,
# }

spec = ModelSpec(matrix, operations)
net = Network(spec, num_labels=10, in_channels=3, stem_out_channels=128, num_stacks=3, num_modules_per_stack=3)
# noqa 401

op2index_201 = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

op2index_darts = {
    'none': 0,
    'skip_connect': 1,
    'sep_conv_3x3': 2,
    'sep_conv_5x5': 3,
    'dil_conv_3x3': 4,
    'dil_conv_5x5': 5,
    'avg_pool_3x3': 6,
    'max_pool_3x3': 7
}

INF = 1000

reward_type2index = {
    'accuracy': -1,
    'ntk': 0,
    'region': 1,
    'mse': 2,
}


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


class Buffer_Reward_Generator(object):
    def __init__(self, xargs, space_name, space_ops, dataset, dataset_val, class_num):
        # self.__super__()
        self.reward_type2index = reward_type2index
        self._reward_types = ["ntk", "region", "mse"]
        # self._reward_types = ["ntk", "mse"]
        # self._reward_types = ["ntk"]
        self._reward_sign = {"ntk": -1, "mse": -1, "region": 1} # ntk/mse: lower the better; region: higher the better
        self._buffers = {key: [] for key in self._reward_types}
        self._buffers_bad = [] # indicator of bad architectures
        self._buffers_change = {key: [] for key in self._reward_types}
        self._buffer_length = getattr(xargs, "te_buffer_size", 10)
        self._xargs = xargs
        self._xargs.init = 'kaiming_norm'
        self._xargs.batch_size = getattr(xargs, "batch_size", 64)
        self._xargs.repeat = getattr(xargs, "repeat", 3)
        self._space_name = space_name
        self._space_ops = space_ops
        self._loader = torch.utils.data.DataLoader(dataset, batch_size=self._xargs.batch_size, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
        self._loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self._xargs.batch_size, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
        self._class_num = class_num
        if space_name == 'nas-bench-101':
            # self._region_model = Linear_Region_Collector(input_size=(64, 3, 32, 32), sample_batch=3,
            #                                              dataset=xargs.dataset, data_path=xargs.data_path,
            #                                              seed=xargs.rand_seed)
            self._region_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3,
                                                         dataset=xargs.dataset, data_path=xargs.data_path,
                                                         seed=xargs.rand_seed)
        elif space_name == 'nas-bench-201':
            self._region_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3,
                                                         dataset=xargs.dataset, data_path=xargs.data_path,
                                                         seed=xargs.rand_seed)
        else:
            self._region_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3,
                                                         dataset=xargs.dataset, data_path=xargs.data_path,
                                                         seed=xargs.rand_seed)

        if space_name == 'nas-bench-101':
            self._model_config = edict({'num_labels': class_num, 'in_channels': 3, 'stem_out_channels': 6,
                                        'num_stacks': 1, 'num_modules_per_stack': 1, 'use_stem': True})
            self._model_config_thin = edict({'num_labels': class_num, 'in_channels': 1, 'stem_out_channels': 6,
                                        'num_stacks': 1, 'num_modules_per_stack': 1, 'use_stem': True})
        elif space_name == 'nas-bench-201':
            self._model_config = edict({'name': 'DARTS-V1', 'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                                        'max_nodes': xargs.max_nodes, 'num_classes': class_num, 'space' : space_ops,
                                        'affine' : True, 'track_running_stats': True,
                                       })
            self._model_config_thin = edict({'name': 'DARTS-V1', 'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                             'max_nodes': xargs.max_nodes, 'num_classes': class_num, 'space' : space_ops,
                                             'affine'   : True, 'track_running_stats': True,
                                            })
        else:
            self._model_config = edict({'name': 'DARTS-V1',
                                        'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                                        'max_nodes': xargs.max_nodes, 'num_classes': class_num, 'space': space_ops,
                                        'imagenet': False,
                                        'affine': True, 'track_running_stats': True,
                                        'super_type': 'nasnet-super', 'steps': 4, 'multiplier': 4,
                                       })
            self._model_config_thin = edict({'name': 'DARTS-V1',
                                             'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                             'max_nodes': xargs.max_nodes, 'num_classes': class_num, 'space' : space_ops,
                                             'imagenet': False,
                                             'affine': True, 'track_running_stats': True,
                                             'super_type': 'nasnet-super', 'steps': 4, 'multiplier': 4,
                                            })
        # prepare supernets with random initialization
        self._networks = []
        self._networks_thin = []
        for _ in range(self._xargs.repeat):
            if space_name == 'nas-bench-101':
                pass
                # spec = ModelSpec(matrix, operations)
                # network_thin = Network(spec, **self._model_config_thin)
            else:
                network = get_cell_based_tiny_net(self._model_config).cuda().train()  # write get network for 101
                init_model(network, xargs.init)
                self._networks.append(network)
                network_thin = get_cell_based_tiny_net(self._model_config_thin).cuda().train()
                init_model(network_thin, xargs.init)
                self._networks_thin.append(network_thin)
        # prepare data samples
        self._ntk_input_data = []
        for i, (inputs, targets) in enumerate(self._loader):
            if i >= self._xargs.repeat: break
            self._ntk_input_data.append((inputs, targets))
        self._ntk_target_data = [] # for NTK kernel regression
        for i, (inputs, targets) in enumerate(self._loader_val):
            if i >= self._xargs.repeat: break
            self._ntk_target_data.append((inputs, targets))

    def _update_bad_cases(self, reward_type, reward):
        # re-set "reward_type" of bad architectures to "reward"
        for _type in self._reward_types:
            for _idx, isbad in enumerate(self._buffers_bad):
                if isbad:
                    self._buffers[_type][_idx] = reward
            for _idx, isbad in enumerate(self._buffers_bad):
                if isbad:
                    self._buffers_change[_type][_idx] = (self._buffers[_type][_idx] - self._buffers[_type][_idx-1]) / (max(self._buffers[_type][max(0, _idx+1-self._buffer_length):_idx+1]) - min(self._buffers[_type][max(0, _idx+1-self._buffer_length):_idx+1]) + 1e-6)
                    if _idx + 1 < len(self._buffers_bad):
                        self._buffers_change[_type][_idx+1] = (self._buffers[_type][_idx+1] - self._buffers[_type][_idx]) / (max(self._buffers[_type][max(0, _idx+2-self._buffer_length):_idx+2]) - min(self._buffers[_type][max(0, _idx+2-self._buffer_length):_idx+2]) + 1e-6)

    def arch_str2mask_201(self, arch_str):
        masks = [torch.ones(6, 5) * (-INF)]
        arch_str_list = np.take(arch_str.split('|'), [1, 3, 4, 6, 7, 8])
        for idx, op in enumerate(arch_str_list):
            masks[0][idx, op2index_201[op.split('~')[0]]] = 0
        return masks

    def arch_parameters2mask(self, arch_parameters: List[torch.Tensor]):
        assert isinstance(arch_parameters, list), f"arch_parameters: {arch_parameters}"
        masks = []
        if self._space_name == 'nas-bench-201':
            for _arch in arch_parameters:
                mask = torch.ones_like(_arch) * (-INF)
                for _idx, edge in enumerate(_arch):
                    mask[_idx][edge.argmax()] = 0
                masks.append(mask)
        elif self._space_name == 'darts':
            for _arch in arch_parameters:
                _arch = torch.nn.functional.softmax(_arch.detach().clone(), -1)
                mask = torch.ones_like(_arch) * (-INF)
                n = 2; start = 0
                for i in range(4):
                    end = start + n
                    edges = sorted(range(i + 2), key=lambda x: -max(_arch[start:end][x][k] for k in range(len(_arch[start:end][x]))))[:2]
                    for edge in edges:
                        # mask[edge+start, _arch[edge+start, 1:].argmax()+1] = 0
                        mask[edge+start, _arch[edge+start].argmax()] = 0
                    start = end; n += 1
                masks.append(mask)
        return masks

    def get_ntk_region_mse(self, xargs, arch_parameters, loader, region_model):
        # arch_parameters now has three dim: cell_type, edge, op
        for _r in range(self._xargs.repeat):
            self._networks[_r].set_alphas(arch_parameters) # only create forward nn.module nasbench101
            self._networks_thin[_r].set_alphas(arch_parameters)

        ntks = [0]; mses = [0]; LRs = [0]
        if  'ntk' in self._reward_types and 'mse' in self._reward_types:
            ntks, mses = get_ntk_n(self._ntk_input_data, self._networks, loader_val=self._ntk_target_data, train_mode=True, num_batch=1, num_classes=self._class_num)
        elif 'ntk' in self._reward_types:
            ntks = get_ntk_n(self._ntk_input_data, self._networks, train_mode=True, num_batch=1, num_classes=self._class_num)
        elif 'mse' in self._reward_types:
            _, mses = get_ntk_n(self._ntk_input_data, self._networks, loader_val=self._ntk_target_data, train_mode=True, num_batch=1, num_classes=self._class_num)
        if 'region' in self._reward_types:
            with torch.no_grad():
                region_model.reinit(models=self._networks_thin, seed=xargs.rand_seed)
                LRs = region_model.forward_batch_sample()
                region_model.clear()
        torch.cuda.empty_cache()
        return {
                "ntk": np.mean(ntks), "region": np.mean(LRs), "mse": np.mean(mses),
                "bad": np.mean(ntks)==-1 or np.mean(mses)==-1 # networks of bad gradients
               }

    def get_ntk_region_mse_101(self, xargs, arch, loader, region_model):
        self._networks = []
        self._networks_thin = []
        # arch_parameters now has three dim: cell_type, edge, op
        matrix, operations = arch
        operations = [index2ops_101[ops] for ops in operations]
        spec = ModelSpec(matrix, operations)
        network = Network(spec, **self._model_config).cuda()
        network_thin = Network(spec, **self._model_config_thin).cuda()
        for _r in range(self._xargs.repeat):
            self._networks.append(network)
            self._networks_thin.append(network_thin)

        ntks = [0]; mses = [0]; LRs = [0]
        if  'ntk' in self._reward_types and 'mse' in self._reward_types:
            ntks, mses = get_ntk_n(self._ntk_input_data, self._networks, loader_val=self._ntk_target_data, train_mode=True, num_batch=1, num_classes=self._class_num)
        elif 'ntk' in self._reward_types:
        # if 'ntk' in self._reward_types:
            ntks = get_ntk_n(self._ntk_input_data, self._networks, train_mode=True, num_batch=1, num_classes=self._class_num)
        elif 'mse' in self._reward_types:
            _, mses = get_ntk_n(self._ntk_input_data, self._networks, loader_val=self._ntk_target_data, train_mode=True, num_batch=1, num_classes=self._class_num)
        if 'region' in self._reward_types:
            with torch.no_grad():
                region_model.reinit(models=self._networks_thin, seed=xargs.rand_seed)
                LRs = region_model.forward_batch_sample()
                region_model.clear()
        torch.cuda.empty_cache()
        return {
                "ntk": np.mean(ntks), "region": np.mean(LRs), "mse": np.mean(mses),
                "bad": np.mean(ntks)==-1 or np.mean(mses)==-1 # networks of bad gradients
               }

    def get_reward(self):
        _reward = 0
        if len(self._buffers[self._reward_types[0]]) <= 1:
            # dummy reward for step 0
            return 0, self.reward_type2index[self._reward_types[0]]
        type_reward = [] # tuples of (type, reward)
        for _type in self._reward_types:
            var = self._buffers_change[_type][-1]
            type_reward.append((self.reward_type2index[_type], self._reward_sign[_type] * var))
        if len(type_reward) > 0:
            _reward = sum([_r for _t, _r in type_reward])
        return _reward

    def _buffer_insert(self, results):
        if len(self._buffers[self._reward_types[0]]) == 0:
            self._buffers_bad.append(results['bad'])
            for _type in self._reward_types:
                self._buffers_change[_type].append(0)
                self._buffers[_type].append(results[_type])
        else:
            if results['bad']:
                # set ntk/mse of bad architecture as worst case in current buffer
                if 'ntk' in self._reward_types: results['ntk'] = max(self._buffers['ntk'])
                if 'mse' in self._reward_types: results['mse'] = max(self._buffers['mse'])
            else:
                if 'ntk' in self._reward_types and results['ntk'] > max(self._buffers['ntk']):
                    self._update_bad_cases('ntk', results['ntk'])
                if 'mse' in self._reward_types and results['mse'] > max(self._buffers['mse']):
                    self._update_bad_cases('mse', results['mse'])
            self._buffers_bad.append(results['bad'])
            for _type in self._reward_types:
                self._buffers[_type].append(results[_type])
                var = (self._buffers[_type][-1] - self._buffers[_type][-2]) / (max(self._buffers[_type][-self._buffer_length:]) - min(self._buffers[_type][-self._buffer_length:]) + 1e-6)
                self._buffers_change[_type].append(var)

    def step(self, arch, mask=True, verbose=False, space_name='nas-bench-201'):
        if space_name == 'nas-bench-101':
            # self._region_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3,
            #                                              dataset=self._xargs.dataset, data_path=self._xargs.data_path,
            #                                              seed=self._xargs.rand_seed)
            results = self.get_ntk_region_mse_101(self._xargs, arch, self._loader, self._region_model)
            # results.update({"mse": 0})
            # results = {
            #     "ntk": np.ones(1), "region": np.ones(1), "mse": np.ones(1),
            #     "bad": np.ones(1)
            #    }
            # torch.cuda.empty_cache()
        else:
            if mask:
                if self._space_name == 'nas-bench-201' and isinstance(arch, str):
                    arch_parameters = self.arch_str2mask_201(arch)
                else:
                    arch_parameters = self.arch_parameters2mask(arch)
            else:
                # e.g. for supernet pruning, not single-path
                arch_parameters = arch
            results = self.get_ntk_region_mse(self._xargs, arch_parameters, self._loader, self._region_model)
        self._buffer_insert(results)
        if verbose:
            print("NTK buffer:", self._buffers['ntk'][-self._buffer_length:])
            print("NTK change buffer:", self._buffers_change['ntk'][-self._buffer_length:])
            print("Regions buffer:", self._buffers['region'][-self._buffer_length:])
            print("Regions change buffer:", self._buffers_change['region'][-self._buffer_length:])
            print("MSE buffer:", self._buffers['mse'][-self._buffer_length:])
            print("MSE change buffer:", self._buffers_change['mse'][-self._buffer_length:])
        reward = self.get_reward()
        # reward larger the better
        # torch.cuda.empty_cache()
        return reward

    def _buffer_rank_best(self):
        # return the index of the best based on rankings over three buffers
        rankings = {}
        buffers_sorted = {}
        rankings_all = []
        for _type in self._reward_types:
            buffers_sorted[_type] = sorted(self._buffers[_type], reverse=self._reward_sign[_type]==1) # by default ascending
            num_samples = len(buffers_sorted[_type])
            rankings[_type] = [ buffers_sorted[_type].index(value) for value in self._buffers[_type] ]
        for _idx in range(num_samples):
            rankings_all.append(sum([ rankings[_type][_idx] for _type in rankings.keys() ]))
        return np.argmin(rankings_all)
