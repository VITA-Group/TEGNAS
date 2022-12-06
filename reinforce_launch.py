
import os
import time
import argparse

# TODO please configure TORCH_HOME and data_paths before running
# TORCH_HOME = "/ssd1/chenwy"  # Path that contains the nas-bench-201 database. If you only want to run on NASNET (i.e. DARTS) search space, then just leave it empty
# data_paths = {
#     "cifar10": "/ssd1/cifar.python",
#     "cifar100": "/ssd1/cifar.python",
#     "ImageNet16-120": "/ssd1/ImageNet16",
#     "imagenet-1k": "/ssd1/chenwy/imagenet_final",
# }

TORCH_HOME = "D:/DATASET/tegnas"  # Path that contains the nas-bench-201 database. If you only want to run on NASNET (i.e. DARTS) search space, then just leave it empty
data_paths = {
    "cifar10": "D:/DATASET/tegnas/cifar.python",
    "cifar100": "D:/DATASET/tegnas/cifar.python",
    "ImageNet16-120": "D:/DATASET/tegnas/cifar.python/ImageNet16",
    "imagenet-1k": "D:/DATASET/tegnas/imagenet_final",
}


parser = argparse.ArgumentParser("TENAS_launch")
parser.add_argument('--gpu', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--space', default='nas-bench-201', type=str, choices=['nas-bench-201', 'darts'], help='which nas search space to use')
parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
args = parser.parse_args()


if args.space == "nas-bench-201":
    args.super_type = "basic"  # type of supernet structure
    args.learning_rate = 0.04
elif args.space == "darts":
    args.super_type = "nasnet-super"
    args.learning_rate = 0.07


# timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))
timestamp = "{:}".format(time.strftime('%m-%d-%Y-%H:%M%p', time.gmtime(time.time())))


core_cmd = "CUDA_VISIBLE_DEVICES={gpuid} OMP_NUM_THREADS=4 python ./reinforce.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--learning_rate {LR} --EMA_momentum 0.9 \
--timestamp {timestamp} \
".format(
    gpuid=args.gpu,
    save_dir="./output/search-cell-{space}/REINFORCE-{dataset}".format(space=args.space, dataset=args.dataset),
    max_nodes=4,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=args.space,
    super_type=args.super_type,
    seed=args.seed,
    LR=args.learning_rate,
    timestamp=timestamp,
)

print(core_cmd)
# os.system(core_cmd)

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./reinforce.py --save_dir ./output/search-cell-nas-bench-201/REINFORCE-cifar10
# --max_nodes 4 --dataset cifar10 --data_path D:/DATASET/tegnas/cifar.python --search_space_name nas-bench-201
# --super_type basic --arch_nas_dataset D:/DATASET/tegnas/NAS-Bench-201-v1_0-e61699.pth --track_running_stats 1
# --workers 0 --rand_seed 0 --learning_rate 0.04 --EMA_momentum 0.9 --timestamp 12-05-2022-22:40PM