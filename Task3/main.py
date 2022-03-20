import argparse
from base64 import encode
from copy import deepcopy
import math
from dataloader import get_cifar10, get_cifar100
from model.wrn  import WideResNet, MACNN
from utils      import Trainer
import torch
import torch.optim as optim
from torch.utils.data   import DataLoader

# imported packages
from LR_schedular import GradualWarmupScheduler
from tqdm import tqdm
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 


def main(args):
    
    if(args.num_workers == -1):
        args.num_workers = os.cpu_count()
    print(f'>>> using {args.num_workers} workers')

    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers)
    unlabeled_loader    = DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers)
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)

    model = MACNN(args).to(device)

    # This for wandb logging
    # wandb = None
    # if (args.wandb):
    #     import wandb
    #     wandb.init(project="NN-project", entity="saarland-uni-eyad")
    #     wandb.config.update(args)

    print(args)

    trainer = Trainer(args, device, model, labeled_loader, unlabeled_loader, test_loader, wandb)
    trainer.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    
    # optimizer parameters
    parser.add_argument("--lr", default=0.001, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")

    # scheduler parameters
    parser.add_argument("--fixed-lr", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument("--min-lr", default=0.000001, type=float,
                        help="The minimum learning rate for the scheduler")
    parser.add_argument("--multiplier", default=3, type=float,
                        help="The constant to multiply the base learning rate by to calculate the highest learning rate.")
    parser.add_argument("--patience", default=10, type=int,
                        help="The patience for the learning rate scheduler")
    parser.add_argument("--factor", default=0.5, type=float,
                        help="The factor for the learning rate scheduler")
    # Training parameters
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--epochs', default=300, type=int, help='The nubmer of total epochs used for training')
    parser.add_argument('--warmup-epochs', type=int, default=0, help="Epochs after which SSL is performed")
    parser.add_argument('--num-workers', default=-1, type=int, help="Number of workers to launch during training")
    parser.add_argument("--att-loss", action="store_true", 
                        help="to compute the attention loss and add it to the total loss.")
    # ssl threshold
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    # model parameters
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--attention-heads", type=int, default=4,
                        help="number of attention heads")
    
    # output parameters
    parser.add_argument("--log-dir", type=str, default="./runs",
                    help="Path to save model's log")
    parser.add_argument("--model-dir", type=str, default="./saved_models",
                    help="Flag to save model")

    # # wandb flag
    # parser.add_argument("--wandb", action="store_true", 
    #                     help="expand labels to fit eval steps")
    
    # Cifar100"
    #   ! python ./main.py --train-batch 250 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 250 --threshold 0.95 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'
    #   ! python ./main.py --train-batch 250 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 250 --threshold 0.75 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'
    #   ! python ./main.py --train-batch 250 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 250 --threshold 0.6 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'
    # 4000:
    #   ! python ./main.py --train-batch 400 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 4000 --threshold 0.95 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'
    #   ! python ./main.py --train-batch 400 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 4000 --threshold 0.75 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'
    #   ! python ./main.py --train-batch 400 --test-batch 1024  --num-workers 4 --warmup-epochs 50 --epochs 200 --num-labeled 4000 --threshold 0.6 --wandb --log-dir '/content/drive/MyDrive/Saarland/NN/Project/runs' --model-dir '/content/drive/MyDrive/Saarland/NN/Project/saved_models'

    # Cifar100"
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 2500  --threshold 0.95 --lr 0.001 --min-lr 0.00000005
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 2500  --threshold 0.75 --lr 0.001 --min-lr 0.00000005
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 2500  --threshold 0.6 --lr 0.001 --min-lr 0.00000005
    # 10000:
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 10000  --threshold 0.95 --lr 0.001 --min-lr 0.00000005
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 10000  --threshold 0.75 --lr 0.001 --min-lr 0.00000005
    # python ./Task1_pseudoLabeling/main.py --dataset cifar100 --train-batch 512 --test-batch 1024 --warmup-epochs 10 --epochs 300 --num-labeled 10000  --threshold 0.6 --lr 0.001 --min-lr 0.00000005
    
    args = parser.parse_args()
    
    s = time.time()
    main(args)
    print("Total time: {}".format(time.time() - s))
    # wandb.finish()
    print("Done. \n\n")