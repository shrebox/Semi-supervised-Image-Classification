from collections import Counter
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data   import DataLoader
from copy import deepcopy
import numpy as np


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
    def __init__(self, args, device, model, labeled_dataset, unlabeled_dataset, test_loader):
        self.args = args
        self.BEST_VAL_LOSS = 1_000_000_000
        self.device = device
        self.ssl_epochs = args.epochs - args.warmup_epochs

        # loaders
        self.labeled_loader = DataLoader(labeled_dataset, 
                                                batch_size = self.args.train_batch, 
                                                shuffle = True, 
                                                num_workers=self.args.num_workers)
        self.unlabeled_loader = DataLoader(unlabeled_dataset, 
                                                batch_size=self.args.train_batch,
                                                shuffle = True, 
                                                num_workers=self.args.num_workers)
        self.test_loader = test_loader

        # datasets
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_dataset = labeled_dataset

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=args.wd, amsgrad=True)
        # self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs, eta_min=self.args.min_lr, last_epoch=-1)
        # self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=self.args.multiplier, total_epoch=self.args.warmup_epochs, after_scheduler=self.cosine_scheduler)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=10,
                                                            eta_min=self.args.lr/50)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.factor, patience=args.patience, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=args.min_lr, eps=1e-08, verbose=True)

        self.iter_per_epoch = len(self.unlabeled_loader)

        if (not os.path.isdir(args.model_dir)):
                os.makedirs(args.model_dir)
        self.MODEL_COMMENT = f'_dataset_{args.dataset}_labeledData_{args.num_labeled}_lr_{args.lr}_threshold_{args.threshold}_batchSize_{args.train_batch}_epochs_{args.epochs}'
        self.writer = SummaryWriter(log_dir = args.log_dir + '/' + self.MODEL_COMMENT) # Writer will output to ./runs/ directory by default    

            
    def epoch_train(self, train_loader, epoch):
        self.model.train()
        train_running_loss, train_running_acc = 0, 0
        num_batches = len(train_loader)
        for x_l, y_l in train_loader:
            x_l, y_l    = x_l.to(self.device), y_l.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x_l)
            loss = self.criterion(output, y_l)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_running_acc += accuracy(output, y_l, topk=(1,))[0].item()
                train_running_loss += self.criterion(output, y_l).item()
        train_loss, train_acc = train_running_loss / num_batches, train_running_acc / num_batches
        print('>>> Epoch {} Train Avg Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, train_loss, train_acc))
        return train_loss, train_acc

    
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
        print('>>> Epoch {} Test Avg Loss: {:.3f}, Test Accuracy: {:.3f}, learning rate: {}'.format(epoch, test_loss, test_acc, self.optimizer.param_groups[0]['lr']))

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


    @torch.no_grad()
    def generate_pseudo_examples(self):
        c = Counter() 
        pseudo_x = []
        pseudo_y = []
        self.model.eval()
        for x_u, y_u in self.unlabeled_loader:
            x_u = x_u.to(self.device)
            y_u = y_u.to(self.device)
            output_ul = self.model(x_u)
            pseudo_labeles          = torch.argmax(output_ul, dim=1) # pseudo labels
            pseudo_probabilities    = torch.softmax(output_ul, dim=1) # pseudo probabilities
            ouptut_ul_inds          = torch.nonzero(pseudo_probabilities > self.args.threshold)
            if (ouptut_ul_inds.nelement() == 0):
                        continue
            x_np = np.moveaxis(x_u[ouptut_ul_inds[:, 0]].detach().cpu().numpy(), 1, -1).astype(np.uint8)
            y_np = pseudo_labeles[ouptut_ul_inds[:, 0]].detach().cpu().numpy()
            c.update(y_np)
            pseudo_x.append(x_np)
            pseudo_y.append(y_np)
        self.model.train()
        print('>>> Generated Pseudo Examples: {}, the distribution {}'.format(len(pseudo_x), c))
        return pseudo_x, pseudo_y

    def ssl_train(self):
        print('>>> SSL Training')
        self.model.train()
        merged_labeled_dataset = deepcopy(self.labeled_dataset)

        merged_dataloader = self.labeled_loader

        for epoch in range(1, self.ssl_epochs+1):

            pseudo_x, pseudo_y = self.generate_pseudo_examples()

            self.model.train()
            if (len(pseudo_x) != 0):
                merged_labeled_dataset.data     =   np.concatenate([self.labeled_dataset.data, *pseudo_x], 0)
                merged_labeled_dataset.targets  =   np.concatenate([self.labeled_dataset.targets, *pseudo_y], 0)
                merged_dataloader      = DataLoader(merged_labeled_dataset, 
                                                batch_size = self.args.train_batch, 
                                                shuffle = True, 
                                                num_workers=self.args.num_workers)

            for x, y in merged_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output  =   self.model(x)
                loss  =   self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

            val_loss, val_acc = self.test(epoch)
            # self.scheduler.step(epoch+self.args.warmup_epochs)
            self.scheduler.step()
            self.metrics_logging(epoch+self.args.warmup_epochs, val_loss, val_acc)

    def warmup_train(self):
        print(">>> Warmup Training")
        self.model.train()
        for epoch in range(1, self.args.warmup_epochs+1):
            train_loss, train_acc   = self.epoch_train(self.labeled_loader, epoch)
            val_loss, val_acc       = self.test(epoch)
            # self.scheduler.step(epoch)
            self.scheduler.step()
            self.metrics_logging(epoch, val_loss, val_acc)
            self.metrics_logging(epoch, train_loss, train_acc, phase='Train')

    def training_loop(self):
        self.warmup_train()
        self.load_best_model()
        # self.reset_lr_schedular(self.args.lr)
        self.ssl_train()
    
    def metrics_logging(self, epoch, val_loss, val_acc, phase="Test"):
        self.writer.add_scalar(f'{phase}/loss', val_loss, epoch)
        self.writer.add_scalar(f'{phase}/accuracy', val_acc, epoch)

    def load_best_model(self):
        print(">>> Loading Best Model", end="-")
        torch.cuda.empty_cache()

        checkpoint = torch.load(os.path.join(self.args.model_dir, f'best_model{self.MODEL_COMMENT}.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer.param_groups[0]['lr'] = lr
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=1e-08, weight_decay=self.args.wd, amsgrad=False)
        # self.BEST_VAL_LOSS = checkpoint['va_loss']
        print("val_loss: {:.3f}".format(self.BEST_VAL_LOSS))
        print(f'Model loaded from epoch {checkpoint["epoch"]}')

    def reset_lr_schedular(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.factor, patience=self.args.patience, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=self.args.min_lr, eps=1e-08, verbose=True)