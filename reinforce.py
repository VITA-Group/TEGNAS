import os, sys, time, random, argparse
from collections import namedtuple
import numpy as np
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger
from procedures   import Buffer_Reward_Generator
from log_utils    import time_string
from nas_201_api  import NASBench201API as API
from models       import CellStructure, get_search_spaces


INF = 1000


# genotype class for darts
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class Policy_DARTS(nn.Module):

    def __init__(self, max_nodes, search_space):
        super(Policy_DARTS, self).__init__()
        self.max_nodes    = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index   = {}
        self._steps = 4
        self._multiplier = 4
        self.edge_keys = []
        for i in range(self._steps):
            for j in range(2+i):
                node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
                self.edge_keys.append(node_str)
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.num_edges  = len(self.edge_keys)
        self._arch_normal = nn.Parameter(1e-3*torch.randn(14, len(search_space)))
        self._arch_reduce = nn.Parameter(1e-3*torch.randn(14, len(search_space)))
        self.arch_parameters = [self._arch_normal, self._arch_reduce]

    def load_arch_params(self, arch_params):
        self.arch_parameters[0].data.copy_(arch_params[0])
        self.arch_parameters[1].data.copy_(arch_params[1])

    # need both arch_parameters (masks) for reward generator; and genotype string for proxy inference
    def generate_arch(self, actions):
        arch_parameters = [-INF*torch.ones_like(alpha) for alpha in self.arch_parameters]
        for cell_idx, action in enumerate(actions):
            for edge_idx, edge in enumerate(action):
                if edge > -1: # edge 0, 1,
                    # arch params is of shape 14*7
                    arch_parameters[cell_idx][edge_idx, edge] = 0
        return arch_parameters # [2][14, 7]

    def genotype(self, weights=None):
        if weights is None:
            # parse policy itself
            weights = self.arch_parameters

        def _parse(weights):
            gene = []
            n = 2; start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                selected_edges = []
                _edge_indice = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.search_space.index('none')))[:2]
                for _edge_index in _edge_indice:
                    _op_indice = list(range(W.shape[1]))
                    _op_indice.remove(self.search_space.index('none'))
                    _op_index = sorted(_op_indice, key=lambda x: -W[_edge_index][x])[0]
                    selected_edges.append( (self.search_space[_op_index], _edge_index) )
                gene += selected_edges
                start = end; n += 1
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return Genotype(normal=gene_normal, normal_concat=[2, 3, 4, 5], reduce=gene_reduce, reduce_concat=[2, 3, 4, 5])

    def forward(self):
        alphas = [nn.functional.softmax(self.arch_parameters[0], dim=-1), nn.functional.softmax(self.arch_parameters[1], dim=-1)]
        return alphas


class Policy(nn.Module):

    def __init__(self, max_nodes, search_space):
        super(Policy, self).__init__()
        self.max_nodes    = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index   = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                self.edge2index[ node_str ] = len(self.edge2index)
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(len(self.edge2index), len(search_space)) )

    def load_arch_params(self, arch_params):
        self.arch_parameters.data.copy_(arch_params)

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name  = self.search_space[ actions[ self.edge2index[ node_str ] ] ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return CellStructure( genotypes )

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[ self.edge2index[node_str] ]
                    op_name = self.search_space[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return CellStructure( genotypes )

    def forward(self):
        alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas

class Policy101(nn.Module):

    def __init__(self, max_nodes, all_archs_numpy):
        super(Policy101, self).__init__()
        self.all_archs_numpy = all_archs_numpy
        # self.max_nodes    = max_nodes
        # self.search_space = deepcopy(search_space)
        # self.edge2index   = {}
        # for i in range(1, max_nodes):
        #     for j in range(i):
        #         node_str = '{:}<-{:}'.format(i, j)
        #         self.edge2index[ node_str ] = len(self.edge2index)
        self.arch_parameters_matrix = nn.Parameter( 1e-3*torch.randn(21, 2) )
        self.arch_parameters_ops = nn.Parameter(1e-3 * torch.randn(5, 4))

    # def load_arch_params(self, arch_params):
    #     self.arch_parameters.data.copy_(arch_params)

    # def generate_arch(self, actions_matrix, actions_ops):
    #     matrix = np.zeros((7,7)).astype(np.int)
    #     k = 0
    #     for z, i in enumerate(range(6, 0, -1)):
    #         matrix[z,z+1:] = actions_matrix[k:k+i].cpu().numpy()
    #         k += i
    #     operations = [0] + [a+1 for a in actions_ops.cpu().numpy()] + [4]
    #     matrix = tuple([tuple(m) for m in matrix])
    #     operations = tuple(operations)
    #     return matrix, operations

    def check_arch(self, actions_matrix, actions_ops):
        matrix = np.zeros((7,7)).astype(np.int)
        k = 0
        for z, i in enumerate(range(6, 0, -1)):
            matrix[z,z+1:] = actions_matrix[k:k+i].cpu().numpy()
            k += i
        ops = [a for a in actions_ops.cpu().numpy()]
        exist_index = [i for (i, o) in enumerate(ops) if o != 0]
        if exist_index == []:
            return False
        operations = np.array([0] + [ops[i] for i in exist_index] + [4])
        exist_index_matrix = [0] + [i+1 for i in exist_index] + [6]
        matrix = matrix[exist_index_matrix, :][:, exist_index_matrix]
        arch = list(np.concatenate((matrix.flatten(), operations)))
        if arch in self.all_archs_numpy:
            matrix = tuple([tuple(m) for m in matrix])
            operations = tuple(operations)
            return matrix, operations
        else:
            return False

    def forward(self):
        alphas_matrix  = nn.functional.softmax(self.arch_parameters_matrix, dim=-1)
        alphas_ops = nn.functional.softmax(self.arch_parameters_ops, dim=-1)
        return alphas_matrix, alphas_ops

def select_action_101(policy):
    action_matrix = torch.zeros(21).type(torch.int64).cuda()
    action_ops = torch.zeros(5).type(torch.int64).cuda()
    probs_matrix, probs_ops = policy()
    m_matrix = Categorical(probs_matrix)
    m_ops = Categorical(probs_ops)
    while True:
        arch = policy.check_arch(action_matrix, action_ops)
        if arch is not False:
            break
        else:
            # probs_matrix, probs_ops = policy()
            # m_matrix = Categorical(probs_matrix)
            # m_ops = Categorical(probs_ops)
            action_matrix = m_matrix.sample()
            action_ops = m_ops.sample()
    # arch = policy.generate_arch(action_matrix, action_ops)
    return arch, m_matrix.log_prob(action_matrix), m_ops.log_prob(action_ops)

    # return (m_matrix.log_prob(action_matrix), action_matrix.cpu().tolist()), (m_ops.log_prob(action_ops), action_ops.cpu().tolist())


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator   = 0
        self._denominator = 0
        self._momentum    = momentum

    def update(self, value):
        self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def select_action(policy):
    probs = policy()
    if len(probs) == 2: # Hardcoded for DARTs
        m = [Categorical(prob) for prob in probs]
        # DARTS, -1 for not using an edge, mute some edges by index_of_action of prob
        actions = [_m.sample() for _m in m]
        for cell_idx, action in enumerate(actions):
            cum_edges = 2
            # start from the 2nd block
            for block_idx in range(1, 4):
                _logit = []
                for edge in range(2+block_idx):
                    _logit.append(policy.arch_parameters[cell_idx][edge+cum_edges, actions[cell_idx][edge+cum_edges]].item())
                selected_edges = np.random.choice(np.arange(2+block_idx)+cum_edges, size=2, replace=False, p=torch.nn.functional.softmax(torch.Tensor(_logit).cuda(), dim=0).detach().cpu().numpy())
                # mute some edges
                for edge in range(2+block_idx):
                    if (edge + cum_edges) not in selected_edges:
                        actions[cell_idx][edge+cum_edges] = -1
                cum_edges += 2+block_idx
        return torch.cat([torch.index_select(_m.log_prob(_action.clamp(0)), 0, torch.where(_action >= 0)[0]) for _m, _action in zip(m, actions)], dim=0), [action.cpu().tolist() for action in actions]
    else:
        # nas-bench-201
        m = Categorical(probs)
        action = m.sample()
        return m.log_prob(action), action.cpu().tolist()


def main(xargs, nas_bench):
    PID = os.getpid()
    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))
    start_time = time.time()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads( xargs.workers )
    prepare_seed(xargs.rand_seed)

    xargs.init = 'kaiming_norm'
    xargs.save_dir = xargs.save_dir + \
        "/LR%.2f-%s-buffer%d-batch%d-repeat%d"%(xargs.learning_rate, xargs.init, xargs.te_buffer_size, xargs.batch_size, xargs.repeat) + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    logger = prepare_logger(xargs)

    if xargs.dataset == 'cifar10':
        dataname = 'cifar10-valid'
    else:
        dataname = xargs.dataset
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    # logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}'.format(xargs.dataset, len(train_data), len(valid_data)))
    # logger.log('||||||| {:10s} |||||||'.format(xargs.dataset))


    if xargs.search_space_name == 'nas-bench-101':
        # search_space = get_search_spaces('cell', xargs.search_space_name)
        all_archs = list(nas_bench.keys())
        all_archs_numpy = [list(np.concatenate((np.array(matrix).flatten(), np.array(ops)))) for matrix, ops in all_archs]
        policy    = Policy101(xargs.max_nodes, all_archs_numpy).cuda()
    elif xargs.search_space_name == 'nas-bench-201':
        search_space = get_search_spaces('cell', xargs.search_space_name)
        policy    = Policy(xargs.max_nodes, search_space).cuda()
    elif xargs.search_space_name == 'darts':
        search_space = get_search_spaces('cell', xargs.search_space_name)
        policy    = Policy_DARTS(xargs.max_nodes, search_space).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=xargs.learning_rate)
    #optimizer = torch.optim.SGD(policy.parameters(), lr=xargs.learning_rate)
    eps       = np.finfo(np.float32).eps.item()
    baseline  = ExponentialMovingAverage(xargs.EMA_momentum)
    logger.log('policy    : {:}'.format(policy))
    logger.log('optimizer : {:}'.format(optimizer))
    logger.log('eps       : {:}'.format(eps))

    # nas dataset load
    # logger.log('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
    arch_enum = set()
    # REINFORCE
    trace = []
    accuracy_history = [] # for 201: save gt accuracy
    start_time = time.time()
    time_proxy_total = 0
    accuracy_best = 0
    accuracy_test_best = 0
    total_steps = 500
    step_current = 0 # for tensorboard
    te_reward_generator = Buffer_Reward_Generator(xargs, xargs.search_space_name, None, train_data, valid_data, class_num)

    for _step in range(total_steps):
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-5:])))
        if xargs.search_space_name == 'nas-bench-101':
            # arch = []
            arch, log_prob_matrix, log_prob_ops = select_action_101(policy)
            # while arch not in all_archs:
            # (log_prob_matrix, action_matrix), (log_prob_ops, action_ops) = select_action_101(policy)
            # while policy.check_arch(action_matrix, action_ops):
            #     (log_prob_matrix, action_matrix), (log_prob_ops, action_ops) = select_action_101(policy)
            #     # print(action_matrix, action_ops)
            #     arch = policy.generate_arch(action_matrix, action_ops)  # CellStructure object for nas-bench-201, arch_params (Tensor) for DARTS
        else:
            log_prob, action = select_action(policy)
        arch_enum.add(arch)
        if xargs.search_space_name == 'nas-bench-101':
            # arch_idx = nas_bench.query_index_by_arch(arch)
            # archinfo = nas_bench.query_meta_info_by_index(arch_idx)
            # accuracy = archinfo.get_metrics(dataname, 'x-valid')['accuracy']
            accuracy = nas_bench[arch]['scratch']['valid']['acc'][108]['avg'] * 100
            accuracy_test = nas_bench[arch]['scratch']['test']['acc'][108]['avg']*100
            if accuracy>accuracy_best: accuracy_best=accuracy
            if accuracy_test>accuracy_test_best: accuracy_test_best=accuracy_test
            accuracy_history.append(accuracy)
            logger.writer.add_scalar("accuracy/search", accuracy, step_current)
            time_elapsed = time.time() - start_time
            num_unique_samples = len(arch_enum)

            reward = te_reward_generator.step(arch, space_name=xargs.search_space_name)
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], step_current)
            logger.writer.add_scalar("TE/Linear_Regions", te_reward_generator._buffers['region'][-1], step_current)
            logger.writer.add_scalar("TE/MSE", te_reward_generator._buffers['mse'][-1], step_current)
            # logger.writer.add_scalar("accuracy/derive", accuracy, step_current)
            # logger.writer.add_scalar("accuracy_test/derive", accuracy_test, step_current)
            probs_matrix, probs_ops = policy()
            logger.writer.add_scalar("reinforce/entropy/matrix", -(torch.log(probs_matrix) * probs_matrix).sum(1).mean(0), step_current)
            logger.writer.add_scalar("reinforce/entropy/ops", -(torch.log(probs_ops) * probs_ops).sum(1).mean(0), step_current)

            logger.writer.add_scalar("accuracy/valid/derive", accuracy, step_current)
            logger.writer.add_scalar("accuracy/test/derive", accuracy_test, step_current)
            logger.writer.add_scalar("accuracy/valid/derive/unique", accuracy, num_unique_samples)
            logger.writer.add_scalar("accuracy/test/derive/unique", accuracy_test, num_unique_samples)
            logger.writer.add_scalar("accuracy/best/valid/derive/unique", accuracy_best, num_unique_samples)
            logger.writer.add_scalar("accuracy/best/test/derive/unique", accuracy_test_best, num_unique_samples)
            logger.writer.add_scalar("accuracy/best/valid/derive/wallclock", accuracy_best, time_elapsed)
            logger.writer.add_scalar("accuracy/best/test/derive/wallclock", accuracy_test_best, time_elapsed)
            # logger.log('step [{:3d}] : accuracy {}'.format(_step, accuracy))
            logger.log(f'Wallclock: {time_elapsed:.3f}, Num: {num_unique_samples}, Step: [{_step:3d}], Valid: {accuracy:.4f} (Best: {accuracy_best:.4f}), Test: {accuracy_test:.4f} (Best: {accuracy_test_best:.4f})')

        elif xargs.search_space_name == 'nas-bench-201':
            arch_idx = nas_bench.query_index_by_arch(arch)
            archinfo = nas_bench.query_meta_info_by_index(arch_idx)
            accuracy = archinfo.get_metrics(dataname, 'x-valid')['accuracy']
            accuracy_history.append(accuracy)
            logger.writer.add_scalar("accuracy/search", accuracy, step_current)
            reward = te_reward_generator.step(nas_bench.query_by_index(arch_idx).arch_str)
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], step_current)
            logger.writer.add_scalar("TE/Linear_Regions", te_reward_generator._buffers['region'][-1], step_current)
            logger.writer.add_scalar("TE/MSE", te_reward_generator._buffers['mse'][-1], step_current)
            logger.writer.add_scalar("accuracy/derive", nas_bench.query_meta_info_by_index(nas_bench.query_index_by_arch(policy.genotype())).get_metrics(dataname, 'x-valid')['accuracy'], step_current)
            probs = policy()
            logger.writer.add_scalar("reinforce/entropy", -(torch.log(probs) * probs).sum(1).mean(0), step_current)
        elif xargs.search_space_name == 'darts':
            genotype = policy.genotype(arch)
            probs = policy()
            reward = te_reward_generator.step(arch)
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], step_current)
            logger.writer.add_scalar("TE/Linear_Regions", te_reward_generator._buffers['region'][-1], step_current)
            logger.writer.add_scalar("TE/MSE", te_reward_generator._buffers['mse'][-1], step_current)
            logger.writer.add_scalar("reinforce/entropy", sum([-(torch.log(prob) * prob).sum(1).mean(0) for prob in probs])/2, step_current)

        logger.writer.add_scalar("reward/reward", reward, step_current)

        trace.append((reward, arch))
        baseline.update(reward)
        # calculate loss
        policy_loss = ( -log_prob_matrix * (reward - baseline.value()) ).sum() + ( -log_prob_ops * (reward - baseline.value()) ).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        step_current += 1
        logger.log('step [{:3d}] : average-reward={:.3f} : policy_loss={:.4f}'.format(_step, baseline.value(), policy_loss.item()))
        # if xargs.search_space_name == 'nas-bench-101':
        #     time_elapsed = time.time()-start_time
        #     num_unique_samples = len(arch_enum)
        #     # logger.log('step [{:3d}] : accuracy {}'.format(_step, accuracy))
        #     logger.log(f'Wallclock: {time_elapsed:.3f}, Num: {num_unique_samples}, Step: [{_step:3d}], Valid: {accuracy:.4f} (Best: {accuracy_best:.4f}), Test: {accuracy_test:.4f} (Best: {accuracy_test_best:.4f})')

        if xargs.search_space_name == 'nas-bench-201':
            arch_idx = nas_bench.query_index_by_arch(policy.genotype())
            archinfo = nas_bench.query_meta_info_by_index(arch_idx)
            accuracy = archinfo.get_metrics(dataname, 'x-valid')['accuracy']
            logger.log('step [{:3d}] : accuracy {}'.format(_step, accuracy))

    end_time = time.time()
    logger.log('REINFORCE finish with {:} steps | time cost {:.1f} s'.format(total_steps, end_time-start_time))

    if xargs.search_space_name == 'nas-bench-101':
        best_idx = te_reward_generator._buffer_rank_best()
        logger.log('101 best of history: {}'.format(accuracy_history[best_idx]))
    elif xargs.search_space_name == 'nas-bench-201':
        best_idx = te_reward_generator._buffer_rank_best()
        logger.log('201 best of history: {}'.format(accuracy_history[best_idx]))

    logger.log('-'*100)

    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reinforce")
    parser.add_argument('--data_path',          type=str,   help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
    parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
    parser.add_argument('--learning_rate',      type=float, help='The learning rate for REINFORCE.')
    parser.add_argument('--EMA_momentum',       type=float, help='The momentum value for EMA.')
    # log
    parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--batch_size',            type=int,   default=64,    help='batch size for ntk')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK, Regions, MSE')
    parser.add_argument('--te_buffer_size',        type=int,   default=10,   help='buffer size for TE reward generator')
    parser.add_argument('--super_type',       type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    # if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    # if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset):
    for args.rand_seed in range(10):
        if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset) or (
                    args.search_space_name != 'nas-bench-201' and args.search_space_name != 'nas-bench-101'):
            nas_bench = None
        else:
            print ('{:} build NAS-Benchmark-API from {:}'.format(time_string(), args.arch_nas_dataset))
            if args.search_space_name == 'nas-bench-101':
                nas_bench = torch.load(args.arch_nas_dataset)
            else:
                nas_bench = API(args.arch_nas_dataset)
        main(args, nas_bench)
