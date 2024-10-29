import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from dataset.CD_dataset import CDDataset
import data_config
# from model.necks.fusion import FDAF
from model import *
from model.losses import *
import sys
import time


def get_loaders(args,root_dir):

    dataConfig = data_config.DataConfig().get_data_config("LEVIR")
    # root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    training_set = CDDataset(root_dir=root_dir, split="train",
                            img_size=args.img_size,is_train=True,
                            label_transform=label_transform)
    val_set = CDDataset(root_dir=root_dir, split="val",
                            img_size=args.img_size,is_train=False,
                            label_transform=label_transform)
    test_set = CDDataset(root_dir=root_dir, split="test",
                            img_size=args.img_size,is_train=False,
                            label_transform=label_transform)

    datasets = {'train': training_set, 'val': val_set, 'test': test_set}
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
                    for x in ['train', 'val',"test"]}

    return dataloaders["train"],dataloaders["val"],dataloaders["test"]


def get_scheduler(optimizer, scheduler,epochs):
    if scheduler == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif scheduler == 'step':
        step_size = epochs//3
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented',scheduler)
    return scheduler

def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

def get_optimizer(args,params):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            # self.net_G.parameters(), 
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4)
        
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr,weight_decay=0)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params, 
            lr=args.lr,
            betas=(0.9, 0.999), 
            weight_decay=0.01
            )

    return optimizer


def get_loss(args,dataloaders):
    weight = torch.tensor([0.2, 0.8]) 
    if args.loss == 'ce':
        loss = cross_entropy
    elif args.loss=="ce2":
        loss = nn.BCEWithLogitsLoss(weight=weight)
    elif args.loss == "miou":
        alpha   = np.asarray(get_alpha(dataloaders.train))
        alpha   = alpha/np.sum(alpha)
        weights = 1-torch.from_numpy(alpha).cuda()
        loss = mIoULoss(weight=weights,size_average=True,n_classes=args.n_class).cuda()
    elif args.loss == "mmiou":
        loss = mmIoULoss(n_classes=args.n_class).cuda()
    else:
        raise NotImplemented(args.loss)
    
    return loss


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        # try:
        with open(self.log_path, mode='a') as f:
            f.write(message)
        # except FileNotFoundError:
        #     with open(self.log_path, mode='w') as f:
        #         f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out
