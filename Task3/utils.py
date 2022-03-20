from collections import Counter
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data   import DataLoader
from model.wrn  import WideResNet
from LR_schedular import GradualWarmupScheduler, get_cosine_schedule_with_warmup
from copy import deepcopy
import numpy as np
import time


def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Trainer():
    def __init__(self, args, device, model, labeled_loader, unlabeled_loader, test_loader, wandb_logger):
        self.args = args
        self.BEST_VAL_LOSS = 1_000_000_000
        self.device = device

        # loaders
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.kl_dis = torch.nn.KLDivLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        if (not args.fixed_lr):
            # self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs, eta_min=self.args.min_lr, last_epoch=-1)
            # self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=self.args.multiplier, total_epoch=self.args.warmup_epochs, after_scheduler=self.cosine_scheduler)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, self.args.epochs)
            # lambda1 = lambda epoch: 0.96 ** epoch
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.factor, patience=args.patience, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=args.min_lr, eps=1e-08, verbose=True)

        self.iter_per_epoch = len(self.unlabeled_loader)

        self.wandb_logger = wandb_logger

        if (not os.path.isdir(args.model_dir)):
                os.makedirs(args.model_dir)
        self.MODEL_COMMENT = f'Task3_dataset_{args.dataset}_labeledData_{args.num_labeled}_lr_{args.lr}_threshold_{args.threshold}_batchSize_{args.train_batch}_epochs_{args.epochs}'
        self.writer = SummaryWriter(log_dir = args.log_dir + '/' + self.MODEL_COMMENT) # Writer will output to ./runs/ directory by default    

    def epoch_train(self, epoch):
        self.model.train()
        labeled_iter = iter(self.labeled_loader)
        print(f'Epoch: {epoch}, lr: {self.optimizer.param_groups[0]["lr"]}', end=", ")
        start = time.time()
        
        for (x_uw, x_us), _ in self.unlabeled_loader:
            try:
                x_l, y_l            =   next(labeled_iter)
            except StopIteration:
                labeled_iter      =   iter(self.labeled_loader)
                x_l, y_l            = next(labeled_iter)
            x_uw, x_us, = x_uw.to(self.device), x_us.to(self.device)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)

            output_l = self.model(x_l)

            output_x_w = self.model(x_uw)
            attention_map_w = self.model.attention_map

            output_x_s = self.model(x_us)
            attention_map_s = self.model.attention_map

            # supervised loss
            loss_l = self.criterion(output_l, y_l)
            # pseudo-labels
            pseudo_probabilities    = torch.softmax(output_x_w, dim=1) # pseudo probabilities
            pseudo_labeles          = torch.argmax(pseudo_probabilities, dim=1) # pseudo labels
            ouptut_ul_inds          = torch.nonzero(pseudo_probabilities >= self.args.threshold)
            attention_dist = 0
            if (ouptut_ul_inds.nelement() == 0):
                loss_u = 0
            else:
                loss_u = self.criterion(output_x_s[ouptut_ul_inds[:,0]], pseudo_labeles[ouptut_ul_inds[:,0]])
                if (self.args.att_loss):
                    attention_map_w_correct = attention_map_w[ouptut_ul_inds[:,0]]
                    pseudo_attention_distribution_w = torch.softmax(attention_map_w_correct.view(attention_map_w_correct.shape[0], -1), dim=1)
                    attention_map_s_correct = attention_map_s[ouptut_ul_inds[:,0]]
                    attention_distribution_s = torch.log_softmax(attention_map_s_correct.view(attention_map_s_correct.shape[0], -1), dim=1)

                    attention_dist = self.kl_dis(attention_distribution_s, pseudo_attention_distribution_w)
            
            total_loss = loss_l + loss_u + attention_dist
            total_loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
        if (not self.args.fixed_lr):
            self.scheduler.step()
        print(f'took {time.time() - start} seconds', end=", ")

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        num_batches = len(self.test_loader)
        size = len(self.test_loader.dataset)
        test_running_loss, test_running_acc = 0, 0
        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            test_running_loss += self.criterion(output, y).item()
            test_running_acc += accuracy(output, y, topk=(1,))[0].item()
        test_loss, test_acc = test_running_loss / num_batches, test_running_acc / num_batches
        print('>>> Epoch {} Test Avg Loss: {:.3f}, Test Accuracy: {:.3f}'.format(epoch, test_loss, test_acc))

        if (test_loss < self.BEST_VAL_LOSS):
            self.BEST_VAL_LOSS = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'va_loss': test_loss,
                'va_accuracy': test_acc
                }, os.path.join(self.args.model_dir, f'best_model{self.MODEL_COMMENT}.pth'))

        return test_loss, test_acc

    def training_loop(self):
        # The main training loop
        for epoch in range(1, self.args.epochs+1):
            self.epoch_train(epoch)
            self.test(epoch)
    
    def metrics_logging(self, epoch, val_loss, val_acc, phase="Test"):
        if (self.args.wandb):
            self.wandb_logger.log({f"{phase}-loss": val_loss, f'{phase}-accuracy': val_acc}, step=epoch)
        self.writer.add_scalar(f'{phase}/loss', val_loss, epoch)
        self.writer.add_scalar(f'{phase}/accuracy', val_acc, epoch)
