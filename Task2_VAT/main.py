import argparse
from base64 import encode
from copy import deepcopy
import math

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy
# from utils import Trainer, model_info
from utils import Trainer
from model.wrn  import WideResNet

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
# TODO delete for submission and server use

warnings.filterwarnings("ignore", category=UserWarning)

def main(args):

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
        
    # assert args.warmup_epochs < args.epochs, "warmup_epochs {} must be less than epoch {}".format(args.warmup_epochs, args.epochs)
    
    # args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    # model_info(model, device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    

    wandb = None
    if (args.wandb):
        import wandb
        wandb.init(project="NN-project", entity="saarland-uni-eyad")
        wandb.config.update(args)

    print(args)

    trainer = Trainer(args, device, model, labeled_dataset,
                      unlabeled_dataset, test_loader, wandb)

    trainer.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    
    # optimizer parameters
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--beta1", default=0.9, type=float,
                        help="beta 1 for adam")
    parser.add_argument("--beta2", default=0.999, type=float,
                        help="beta 2 for adam")
    
    # scheduler parameters
    parser.add_argument("--min-lr", default=0.0000001, type=float,
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
                       
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.05)')

    parser.add_argument('--epochs', default=350, type=int,
                        help='The nubmer of total epochs used for training')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help="Epochs after which SSL is performed")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    
    # ssl threshold
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    
    # model parameters
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    
    # output parameters
    parser.add_argument("--log-dir", type=str, default="./runs",
                    help="Path to save model's log")
    parser.add_argument("--model-dir", type=str, default="./saved_models",
                    help="Flag to save model")

    # SSL loss weighing arguments
    parser.add_argument("--vat_xi", default=10.0, type=float,
                        help="VAT xi parameter")
    parser.add_argument("--vat_eps", default=10.0, type=float,
                        help="VAT epsilon parameter")
    parser.add_argument("--vat_iter", default=1, type=int,
                        help="VAT iteration parameter")
    parser.add_argument("--use_entmin", type=bool, default=False,
                        help="Additional cost of conditional entropy minimization to loss")

    # wandb flag
    parser.add_argument("--wandb", action="store_true",
                        help="expand labels to fit eval steps")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)
    # wandb.finish()
    print("Done ##############################################")


# SOME NOTES
# for labeled data with size 250 set the batch size to 250
# for labeled data with more than 250 set the batch size to 512
# set warmup-epochs to 35 and epochs to 300
