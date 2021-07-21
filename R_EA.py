import os, sys, time, random, argparse
import math
from collections import namedtuple
import numpy as np, collections
from copy import deepcopy
import torch
from pathlib import Path
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
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
DARTS_SPACE           = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']


class Model(object):

    def __init__(self):
        self.arch = None # 201: CellStructure, darts: Genotype
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{:}'.format(self.arch)


def random_architecture_func_201(max_nodes, op_names):
    # return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name  = random.choice( op_names )
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return CellStructure( genotypes )
    return random_architecture


def random_architecture_func_darts(op_names):
    # return a random architecture
    def random_architecture():
        weights = [torch.rand(14, 8).cuda(), torch.rand(14, 8).cuda()]
        return genotype_darts(weights, op_names)
    return random_architecture


def mutate_arch_func_201(op_names):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """
    def mutate_arch_func(parent_arch):
        child_arch = deepcopy( parent_arch )
        node_id = random.randint(0, len(child_arch.nodes)-1)
        node_info = list( child_arch.nodes[node_id] )
        snode_id = random.randint(0, len(node_info)-1)
        xop = random.choice( op_names )
        while xop == node_info[snode_id][0]:
            xop = random.choice( op_names )
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple( node_info )
        return child_arch
    return mutate_arch_func


def mutate_arch_func_darts(op_names):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """
    def mutate_arch_func(parent_geno):
        # parent_arch is darts genotype
        # random.randint are inclusive on two sides
        topology_or_op = random.randint(0, 1) # 0 for topology, 1 for op
        cell_idx = random.randint(0, 1) # 0 for normal, 1 for reduce
        node_idx = random.randint(0, 3)
        input_idx = random.randint(0, 1)
        child_geno = deepcopy( parent_geno )
        if topology_or_op == 0:
            # mutate topology: switch one in-connect from one edge of one node to another in-connect
            input_from_idx = random.randint(0, node_idx+2-1)
            if cell_idx == 0:
                edge = child_geno.normal[node_idx*2+input_idx]
                child_geno.normal[node_idx*2+input_idx] = (edge[0], input_from_idx)
            elif cell_idx == 1:
                edge = child_geno.reduce[node_idx*2+input_idx]
                child_geno.reduce[node_idx*2+input_idx] = (edge[0], input_from_idx)
        elif topology_or_op == 1:
            if cell_idx == 0:
                edge = child_geno.normal[node_idx*2+input_idx]
                _op_names = list(op_names); _op_names.remove(edge[0])
                xop = random.choice( _op_names )
                child_geno.normal[node_idx*2+input_idx] = (xop, edge[1])
            elif cell_idx == 1:
                edge = child_geno.reduce[node_idx*2+input_idx]
                _op_names = list(op_names); _op_names.remove(edge[0])
                xop = random.choice( _op_names )
                child_geno.reduce[node_idx*2+input_idx] = (xop, edge[1])
        return child_geno
    return mutate_arch_func


def genotype_darts(weights, search_space=DARTS_SPACE):
    edge2index   = {}
    _steps = 4
    edge_keys = []
    for i in range(_steps):
        for j in range(2+i):
            node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
            edge_keys.append(node_str)
    edge2index = {key:i for i, key in enumerate(edge_keys)}
    def _parse(weights):
        gene = []
        n = 2; start = 0
        for i in range(_steps):
            end = start + n
            W = weights[start:end].copy()
            selected_edges = []
            _edge_indice = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != search_space.index('none')))[:2]
            for _edge_index in _edge_indice:
                _op_indice = list(range(W.shape[1]))
                _op_indice.remove(search_space.index('none'))
                _op_index = sorted(_op_indice, key=lambda x: -W[_edge_index][x])[0]
                selected_edges.append( (search_space[_op_index], _edge_index) )
            gene += selected_edges
            start = end; n += 1
        return gene
    with torch.no_grad():
        gene_normal = _parse(torch.softmax(weights[0], dim=-1).cpu().numpy())
        gene_reduce = _parse(torch.softmax(weights[1], dim=-1).cpu().numpy())
    return Genotype(normal=gene_normal, normal_concat=[2, 3, 4, 5], reduce=gene_reduce, reduce_concat=[2, 3, 4, 5])


def genotype2mask_201(genotype, op_names=NAS_BENCH_201):
    # genotype is CellStructure
    masks = torch.ones(6, 5).cuda() * (-INF)
    ops = []
    for node in genotype.nodes:
        for edge in node:
            ops.append(op_names.index(edge[0]))
    for idx, op in enumerate(ops):
        masks[idx, op] = 0
    return masks


def arch_distance_201(arch1, arch2, op_names=NAS_BENCH_201):
    # arch is CellStructure
    ops1 = []
    for node in arch1.nodes:
        for edge in node:
            ops1.append(op_names.index(edge[0]))
    ops2 = []
    for node in arch2.nodes:
        for edge in node:
            ops2.append(op_names.index(edge[0]))
    distance = 0
    for op1, op2 in zip(ops1, ops2):
        if op1 != op2:
            distance += 1
    return distance


def genotype2mask_darts(genotype, op_names=DARTS_SPACE):
    masks = [torch.ones(14, 8).cuda() * (-INF), torch.ones(14, 8).cuda() * (-INF)]
    for node_idx in range(4):
        prev_edges = sum([_node_idx+2 for _node_idx in range(node_idx)])
        edges = list(range(prev_edges, prev_edges+node_idx+2))
        edges_active_normal = genotype.normal[node_idx*2:(node_idx+1)*2] # op_str: input_from
        edges_active_reduce = genotype.reduce[node_idx*2:(node_idx+1)*2] # op_str: input_from
        for op_str, input_from in edges_active_normal:
            masks[0][input_from+prev_edges, op_names.index(op_str)] = 0
        for op_str, input_from in edges_active_reduce:
            masks[1][input_from+prev_edges, op_names.index(op_str)] = 0
    return masks


def arch_distance_darts(genotype1, genotype2, op_names=DARTS_SPACE):
    distance = 0
    for node_idx in range(4):
        edges_normal_1 = genotype1.normal[node_idx*2:(node_idx+1)*2] # op_str: input_from
        edges_normal_2 = genotype2.normal[node_idx*2:(node_idx+1)*2] # op_str: input_from
        for edge in edges_normal_1:
            if edge not in edges_normal_2:
                distance += 1
        edges_reduce_1 = genotype1.reduce[node_idx*2:(node_idx+1)*2] # op_str: input_from
        edges_reduce_2 = genotype2.reduce[node_idx*2:(node_idx+1)*2] # op_str: input_from
        for edge in edges_reduce_1:
            if edge not in edges_reduce_2:
                distance += 1
    return distance


def proxy_inference(xargs, arch, nas_bench, logger, step_current, dataname, te_reward_generator=None):
    accuracy = -1 # GT accuracy of arch
    reward_type = 'accuracy'
    if xargs.search_space_name == 'nas-bench-201':
        arch_idx = nas_bench.query_index_by_arch(arch)
        archinfo = nas_bench.query_meta_info_by_index(arch_idx)
        accuracy = archinfo.get_metrics(dataname, 'x-valid')['accuracy']
        if step_current >= 0: logger.writer.add_scalar("accuracy/search", accuracy, step_current)
        start_time = time.time()
        _ = te_reward_generator.step(nas_bench.query_by_index(arch_idx).arch_str)
        if len(te_reward_generator._buffers['ntk']) == 0:
            reward = {'ntk': math.inf, 'region': 0, 'mse': math.inf}
        else:
            reward = {'ntk': -te_reward_generator._buffers['ntk'][-1], 'region': te_reward_generator._buffers['region'][-1], 'mse': -te_reward_generator._buffers['mse'][-1]}
        time_spent = time.time() - start_time
        if step_current >= 0:
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], step_current)
            logger.writer.add_scalar("TE/Linear_Regions", te_reward_generator._buffers['region'][-1], step_current)
            logger.writer.add_scalar("TE/MSE", te_reward_generator._buffers['mse'][-1], step_current)
    elif xargs.search_space_name == 'darts':
        start_time = time.time()
        _ = te_reward_generator.step(genotype2mask_darts(arch))
        if len(te_reward_generator._buffers['ntk']) == 0:
            reward = {'ntk': math.inf, 'region': 0, 'mse': math.inf}
        else:
            reward = {'ntk': -te_reward_generator._buffers['ntk'][-1], 'region': te_reward_generator._buffers['region'][-1], 'mse': -te_reward_generator._buffers['mse'][-1]}
        if step_current >= 0:
            logger.writer.add_scalar("TE/NTK", te_reward_generator._buffers['ntk'][-1], step_current)
            logger.writer.add_scalar("TE/Linear_Regions", te_reward_generator._buffers['region'][-1], step_current)
            logger.writer.add_scalar("TE/MSE", te_reward_generator._buffers['mse'][-1], step_current)
        time_spent = time.time() - start_time
    return reward, accuracy, time_spent


def __best_of_sample(samples, topk=None, sort=False):
    assert len(samples) > 0
    reward_keys = list(samples[0].accuracy.keys())
    if len(reward_keys) == 1:
        reward_type = reward_keys[0]
    else:
        all_rewards = {key: [] for key in reward_keys}
        for sample in samples:
            for key, value in sample.accuracy.items():
                all_rewards[key].append(value)
        for key in reward_keys:
            # get changing range
            if topk is None:
                values = all_rewards[key]
            else:
                assert isinstance(topk, int)
                if sort:
                    values = sorted(all_rewards[key], reverse=True) # descending
                else:
                    values = all_rewards[key]
            values = np.absolute(np.array(values))
            all_rewards[key] = (max(values[:topk]) - min(values[:topk])) / (max(values) - min(values))  # range of change
        reward_type = max(reward_keys, key=lambda i: all_rewards[i])
    parent = max(samples, key=lambda i: i.accuracy[reward_type])
    return parent, reward_type


def best_of_sample(samples, topk=None, sort=False):
    assert len(samples) > 0
    reward_keys = list(samples[0].accuracy.keys())
    if len(reward_keys) == 1:
        reward_type = reward_keys[0]
        parent = max(samples, key=lambda i: i.accuracy[reward_type])
    else:
        all_rewards = {key: [] for key in reward_keys}
        for sample in samples:
            for key, value in sample.accuracy.items():
                all_rewards[key].append(value)
        for key in reward_keys:
            all_rewards[key] = sorted(all_rewards[key], reverse=True) # descending
        rankings = []
        for sample in samples:
            ranking = 0
            for key in reward_keys:
                ranking += all_rewards[key].index(sample.accuracy[key])
            rankings.append((sample, ranking))
        rankings = sorted(rankings, key=lambda i: i[1]) # ascending
        parent = rankings[0][0]
        reward_type = reward_keys[np.argmin([all_rewards[key].index(parent.accuracy[key]) for key in reward_keys])]
    return parent, reward_type


def population_diversity(population, space_name):
    distances = 0
    for p1 in range(len(population)):
        for p2 in range(p1+1, len(population)):
            model1 = population[p1]
            model2 = population[p2]
            if space_name == 'nas-bench-201':
                distances += arch_distance_201(model1.arch, model2.arch)
            elif space_name == 'darts':
                distances += arch_distance_darts(model1.arch, model2.arch)
    return distances / (len(population) * (len(population) - 1) / 2)


def regularized_evolution(xargs, total_steps, step_current, sample_size, random_arch, mutate_arch, logger, PID,
                          history, population, nas_bench, dataname, te_reward_generator=None
                         ):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
        cycles: the number of cycles the algorithm should run for.
        population_size: the number of individuals to keep in the population.
        sample_size: the number of individuals that should participate in each tournament.
        time_budget: the upper bound of searching cost

    Returns:
        history: a list of `Model` instances, representing all the models computed
                during the evolution experiment.
    """

    # Carry out evolution in cycles. Each cycle produces a model and removes another.
    time_cost_training = 0
    for _step in range(total_steps):
        best_arch, _ = best_of_sample(population, topk=2, sort=True)  # choose parent based on changing range out of samples

        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-5:])))

        # Sample randomly chosen models from the current population.
        sample = np.random.choice(list(population), size=sample_size, replace=False)

        # The parent is the best model in the sample.
        parent, reward_type = best_of_sample(sample, topk=2, sort=True)  # choose parent based on changing range out of samples

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy, accuracy_gt, time_cost = proxy_inference(xargs, child.arch, nas_bench, logger, step_current, dataname, te_reward_generator)
        time_cost_training += time_cost
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
        step_current += 1
        logger.writer.add_scalar("evolution/population_diversity", population_diversity(population, xargs.search_space_name), step_current)

        best_arch, best_type = best_of_sample(population, topk=2, sort=True)  # choose parent based on changing range out of samples
        logger.log('step [{:3d}] : best of populuation {:} type={:} : {:}'.format(_step, best_arch.accuracy[best_type], best_type, best_arch.arch))
        if xargs.search_space_name == 'nas-bench-201':
            logger.log('step [{:3d}] => accuracy of population {}'.format(_step, nas_bench.query_meta_info_by_index(nas_bench.query_index_by_arch(best_arch.arch)).get_metrics(dataname, 'x-valid')['accuracy']))
            logger.writer.add_scalar("accuracy/derive", nas_bench.query_meta_info_by_index(nas_bench.query_index_by_arch(best_arch.arch)).get_metrics(dataname, 'x-valid')['accuracy'], step_current)
    return population, history, step_current, time_cost_training


def main(xargs, nas_bench):
    PID = os.getpid()
    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads( xargs.workers )
    prepare_seed(xargs.rand_seed)

    xargs.init = 'kaiming_norm'

    xargs.te_buffer_size = max(xargs.te_buffer_size, xargs.ea_population_size)
    xargs.save_dir = xargs.save_dir + \
        "/sample%d-population%d-%s-steps%d-ntk.regions.mse-buffer%d-batch%d-repeat%d"%(xargs.ea_sample_size, xargs.ea_population_size, xargs.init, xargs.total_steps, xargs.te_buffer_size, xargs.batch_size, xargs.repeat) + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    logger = prepare_logger(xargs)

    if xargs.dataset == 'cifar10':
        dataname = 'cifar10-valid'
    else:
        dataname = xargs.dataset
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}'.format(xargs.dataset, len(train_data), len(valid_data)))
    logger.log('||||||| {:10s} |||||||'.format(xargs.dataset))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.search_space_name == 'nas-bench-201':
        random_arch = random_architecture_func_201(xargs.max_nodes, search_space)
        mutate_arch = mutate_arch_func_201(search_space)
    elif xargs.search_space_name == 'darts':
        random_arch = random_architecture_func_darts(search_space)
        mutate_arch = mutate_arch_func_darts(search_space)
    x_start_time = time.time()
    logger.log('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
    logger.log('-'*30 + ' start searching')

    start_time = time.time()
    step_current = 0 # for tensorboard

    te_reward_generator = Buffer_Reward_Generator(xargs, xargs.search_space_name, search_space, train_data, valid_data, class_num)

    time_cost_training = 0
    ########### Evolution Preparation
    population = collections.deque()
    history = []  # Not used, history of all samples
    # Initialize the population with random models.
    while len(population) < xargs.ea_population_size:
        print("<< ============== JOB (PID = %d) %s [Init population %d/%d] ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-5:]), len(population), xargs.ea_population_size))
        model = Model()
        model.arch = random_arch()
        model.accuracy, _, _time_cost_training = proxy_inference(xargs, model.arch, nas_bench, logger, -1, dataname, te_reward_generator)
        time_cost_training += _time_cost_training
        population.append(model)
        history.append(model)
    #################################

    population, history, step_current, _time_cost_training = regularized_evolution(xargs, xargs.total_steps, step_current, xargs.ea_sample_size, random_arch, mutate_arch, logger, PID,
                                                                                   history, population, nas_bench, dataname, te_reward_generator
                                                                                  )
    time_cost_training += _time_cost_training

    total_time_cost = time.time() - start_time
    logger.log('{:} regularized_evolution finish with {:.1f} s.'.format(time_string(), total_time_cost))
    # best_arch = max(population, key=lambda i: i.accuracy)
    # best_arch, _ = best_of_sample(population, topk = te_reward_generator._buffer_size if te_reward_generator is not None else None)  # choose parent based on changing range out of samples
    best_arch, _ = best_of_sample(population, topk=2, sort=True)  # choose parent based on changing range out of samples
    best_arch = best_arch.arch
    logger.log('{:} best arch is {:}'.format(time_string(), best_arch))

    logger.log('-'*100)
    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
    parser.add_argument('--data_path',          type=str,   help='Path to dataset')
    parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
    parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
    parser.add_argument('--ea_population_size',      type=int,   help='The population size in EA.')
    parser.add_argument('--ea_sample_size',     type=int,   help='The sample size in EA.')
    parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--batch_size',            type=int,   default=64,    help='batch size for ntk')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK, Regions, MSE')
    parser.add_argument('--total_steps',        type=int,   default=500,   help='number of samplings for search')
    parser.add_argument('--te_buffer_size',        type=int,   default=10,   help='buffer size for TE reward generator')
    parser.add_argument('--super_type',       type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset) or args.search_space_name != 'nas-bench-201':
        nas_bench = None
    else:
        print ('{:} build NAS-Benchmark-API from {:}'.format(time_string(), args.arch_nas_dataset))
        nas_bench = API(args.arch_nas_dataset)
    main(args, nas_bench)
