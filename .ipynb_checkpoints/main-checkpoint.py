#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from __future__ import print_function
import os
import cv2
import tqdm
import json
import math
import time
import torch
import Utils
import models
import logging
import datetime
import argparse
import utils_seg
import dataloaders
import numpy as np
import torch.optim

from PIL import Image
from tqdm import tqdm
from imageio import imwrite
from utils_seg import Logger
from utils_seg import helpers
from utils_seg.losses import *
# from dataloaders.voc import VOC
from dataloaders.voc1 import VOC
from base import DataPrefetcher
from torchvision import transforms
from torchvision.utils import make_grid
from models.pyramid_fusion import PyFuse
from utils_seg.helpers import colorize_mask
from torch.utils.tensorboard import SummaryWriter
from utils_seg import transforms as local_transforms
from utils_seg.metrics import eval_metrics, AverageMeter
import pdb
# from azureml.tensorboard import Tensorboard
import warnings
warnings.filterwarnings("ignore")

# %%


parser = argparse.ArgumentParser(description='BiFuse script for 360 Semantic Segmentation!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', default='', type=str, help='Path of source images')
parser.add_argument('--batch_size', default= 16, type=int, help='batch size')
parser.add_argument('--checkpoint_dir', default=None, type=str, help='Path to the saving .pth model')
parser.add_argument('--log_dir', default=None, type=str, help='Path to the saving .pth model')
parser.add_argument('-resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
parser.add_argument('--d', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--val', default = True, type = bool, help = 'Perform validation or not')
<<<<<<< HEAD
parser.add_argument('--epoch', default = 10, type = int, help = 'No of epochs')
=======
<<<<<<< HEAD
parser.add_argument('--epoch', default = 10, type = int, help = 'No of epochs')
=======
parser.add_argument('--epoch', default = 100, type = int, help = 'No of epochs')
>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
parser.add_argument('--momentum', default = 0.99, type = float, help = 'Momentum')
parser.add_argument('--beta', default = 0.01, type = float, help = 'Beta')
parser.add_argument('--weight_decay', default = 10e-5, type = float,help = 'weight_decay')
parser.add_argument('--lr', default = 0.001, type = float, help = 'learning_rate')
parser.add_argument('--val_per_epoch', default = 10, type = int, help = 'validation per epoch')
parser.add_argument('--early_stop', default = 10, type = int, help = 'early stop')
parser.add_argument('--monitor', default = True, type = bool, help = 'monitor params for early stop')
parser.add_argument('--n_gpu', default = 1, type = int, help = 'n_gpu')
parser.add_argument('--name', default = "pyfuse", type = str, help = 'name')
parser.add_argument('--log_step', default = 10, type = int, help = 'log_steps')
parser.add_argument('--num_classes', default = 21, type = int, help = 'Number of classes')
args = parser.parse_args()


# %%


# class MyData(data.Dataset):
#     def __init__(self, root):
#         imgs = os.listdir(root)
#         self.imgs = [os.path.join(root, k) for k in imgs]
#         self.transforms = transforms.Compose([
#             transforms.ToTensor()
#             ])

#     def __getitem__(self, index):
#         img_path = self.imgs[index]
#         rgb_img = Image.open(img_path).convert("RGB")
#         rgb_img = np.array(rgb_img, np.float32) / 255
#         rgb_img = cv2.resize(rgb_img, (1024, 512), interpolation=cv2.INTER_AREA)
#         data = self.transforms(rgb_img)

#         return data

#     def __len__(self):
#         return len(self.imgs)


# %%


def get_available_devices(logger, n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
#         logger.warning('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu

    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    print(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
    available_gpus = list(range(n_gpu))
    return device, available_gpus


# %%

def main(args):
    
    logger = Logger()
#     train_loader = VOC(args.path, args.batch_size, 'train')
    train_loader = VOC("train")
    print(train_loader)
#     val_loader = VOC(args.path, args.batch_size, 'val')
    val_loader = VOC("val")
    device, available_gpus = get_available_devices(logger, args.n_gpu)
    model = PyFuse(50)
#     print(f'\n{model}\n')
    model = torch.nn.DataParallel(model, available_gpus)
    model.to(device)
<<<<<<< HEAD
    loss = FocalLoss()
=======
<<<<<<< HEAD
    loss = FocalLoss()
=======
    loss = CrossEntropyLoss2d()
>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
    print(loss)
    # OPTIMIZER
    optim_params = [
    {'params': model.parameters(), 'lr': args.lr},
    ]

    optimizer = torch.optim.Adam(optim_params,
                             betas=(args.momentum, args.beta),
                             weight_decay=(args.weight_decay))
    
    wrt_mode, wrt_step = 'train_', 0
        
       
        
#     TRANSORMS FOR VISUALIZATION
#     restore_transform = transforms.Compose([
#         local_transforms.DeNormalize(train_loader.MEAN, train_loader.STD),
#         transforms.ToPILImage()])
#     viz_transform = transforms.Compose([
#         transforms.Resize((400, 400)),
#         transforms.ToTensor()])
    
#     train_loader = restore_transform(train_loader)
#     val_loader = restore_transform(val_loader)
#     prefetch = True
#     if device ==  torch.device('cpu'): 
#         prefetch = False
#     if prefetch:
#         train_loader = DataPrefetcher(train_loader, device=device)
#         val_loader = DataPrefetcher(val_loader, device=device)

    torch.backends.cudnn.benchmark = True
    
    monitor = "max Mean_IoU"
        
    # MONITORING
    if args.monitor == 'False':
        mnt_mode = 'off'
        mnt_best = 0
    else:
        mnt_mode, mnt_metric = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = -math.inf if mnt_mode == 'max' else math.inf
        early_stoping = args.early_stop

    # CHECKPOINTS & TENSOBOARD
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.name, start_time)
    helpers.dir_exists(checkpoint_dir)
    config_save_path = os.path.join(checkpoint_dir, 'config.json')
#     with open(config_save_path, 'w') as handle:
#         json.dump(config, handle, indent=4, sort_keys=True)

#     writer_dir = os.path.join(args.log_dir, args.name, start_time)
#     writer = SummaryWriter(writer_dir)

    if args.resume: _resume_checkpoint(args.resume)
    start_epoch = 1
    epoch=args.epoch
    num_classes = args.num_classes
    for epoch in range(start_epoch, epoch+1):
        # RUN TRAIN (AND VAL)
        #pdb.set_trace()
<<<<<<< HEAD
        results = _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer, loss)
=======
<<<<<<< HEAD
        results = _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer, loss)
=======
        results = _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer)
>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
        if do_validation and epoch % args.val_per_epoch == 0:
            results = _valid_epoch(model, epoch)

            # LOGGING INFO
            logger.info(f'\n         ## Info for epoch {epoch} ## ')
            for k, v in results.items():
                logger.info(f'         {str(k):15s}: {v}')

        if train_logger is not None:
            log = {'epoch' : epoch, **results}
            train_logger.add_entry(log)

        # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
        if mnt_mode != 'off' and epoch % args.val_per_epoch == 0:
            try:
                if mnt_mode == 'min': improved = (log[mnt_metric] < mnt_best)
                else: improved = (log[mnt_metric] > mnt_best)
            except KeyError:
                logger.warning(f'The metrics being tracked ({mnt_metric}) has not been calculated. Training stops.')
                break

            if improved:
                mnt_best = log[mnt_metric]
                not_improved_count = 0
            else:
                not_improved_count += 1

            if not_improved_count > early_stoping:
                logger.info(f'\nPerformance didn\'t improve for {early_stoping} epochs')
                logger.warning('Training Stoped')
                break

        # SAVE CHECKPOINT
        if epoch % args.save_period == 0:
            _save_checkpoint(epoch, save_best=improved)


# %%


def _save_checkpoint(epoch, save_best=False):
    state = {
        'arch': type(self.model).__name__,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    logger.info(f'\nSaving a checkpoint: {filename} ...') 
    torch.save(state, filename)

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(state, filename)
        self.logger.info("Saving current best: best_model.pth")


# %%


def _resume_checkpoint(resume_path):
    logger.info(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    start_epoch = checkpoint['epoch'] + 1
    mnt_best = checkpoint['monitor_best']
    not_improved_count = 0

#     if checkpoint['config']['arch'] != config['arch']:
#         self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
#     self.model.load_state_dict(checkpoint['state_dict'])

#     if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
#         self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
#     self.optimizer.load_state_dict(checkpoint['optimizer'])
    # if self.lr_scheduler:
    #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    train_logger = checkpoint['logger']
    logger.info(f'Checkpoint <{resume_path}> (epoch {start_epoch}) was loaded')


# %%
<<<<<<< HEAD
def _reset_metrics():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0


# %%
def _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer, loss):
#     logger.info('\n')
=======


<<<<<<< HEAD
def _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer, loss):
#     logger.info('\n')
=======
def _train_epoch(model,epoch, num_classes, train_loader, logger, optimizer):
#     logger.info('\n')

>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
    model.train()
    wrt_mode = 'train'

    tic = time.time()
<<<<<<< HEAD
#     _reset_metrics()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    tbar = tqdm(train_loader, ncols=130)
    date_time = datetime.date.today()
    for batch_idx, (data, target) in enumerate(tbar):
        data_time.update(time.time() - tic)
#         data, target = data.to(self.device), target.to(self.device)
=======
    _reset_metrics()
    tbar = tqdm(train_loader, ncols=130)
    date_time = datetime.date.today()
    for batch_idx, (data, target) in enumerate(tbar):
        #data_time.update(time.time() - tic)
        #data, target = data.to(self.device), target.to(self.device)
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
#         lr_scheduler.step(epoch=epoch-1)

        # LOSS & OPTIMIZE
        optimizer.zero_grad()
        output = model(data)
<<<<<<< HEAD
        print(batch_idx)
        assert output.size()[2:] == target.size()[1:]
        assert output.size()[1] == num_classes 
        target = ((target).type(torch.int64)).cuda()
        output = output.cuda()
        print("Loss Fn Output Size: ", output.shape)
        print("Loss Fn Target Size: ", target.shape)
#         pdb.set_trace()
=======
<<<<<<< HEAD
        print(batch_idx)
        assert output.size()[2:] == target.size()[1:]
        assert output.size()[1] == num_classes 
        target = ((target).type(torch.int64)).cpu()
        output = output.cpu()
        print("Loss Fn Output Size: ", output.shape)
        print("Loss Fn Target Size: ", target.shape)
        pdb.set_trace()
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
        Loss = loss(output, target)
#         Loss += loss(output[0], target) * 0.4
        

        if isinstance(loss, torch.nn.DataParallel):
            Loss = loss.mean()
        Loss.backward()
        optimizer.step()
<<<<<<< HEAD
        total_loss.update(Loss.item())
=======
        total_Loss.update(Loss.item())
=======
#         assert output[0].size()[2:] == target.size()[1:]
#         assert output[0].size()[1] == num_classes 
        loss = loss(output[0], target)
        loss += loss(output[1], target) * 0.4
        output = output[0]

        if isinstance(loss, torch.nn.DataParallel):
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # LOGGING & TENSORBOARD
<<<<<<< HEAD
#         if batch_idx % log_step == 0:
#             wrt_step = (epoch - 1) * len(train_loader) + batch_idx
=======
        if batch_idx % log_step == 0:
            wrt_step = (epoch - 1) * len(train_loader) + batch_idx
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
#             writer.add_scalar(f'{wrt_mode}/loss', loss.item(), wrt_step)

        # FOR EVAL
        seg_metrics = eval_metrics(output, target, num_classes)
<<<<<<< HEAD
        _update_seg_metrics(*seg_metrics, total_correct, total_label, total_inter, total_union)
        pixAcc, mIoU, _ = _get_seg_metrics(total_correct, total_label, total_inter, total_union).values()
=======
        _update_seg_metrics(*seg_metrics)
        pixAcc, mIoU, _ = _get_seg_metrics().values()
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6

        # PRINT INFO
        tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                            epoch, total_loss.average, 
                                            pixAcc, mIoU,
                                            batch_time.average, data_time.average))

    # METRICS TO TENSORBOARD
#     seg_metrics = _get_seg_metrics()
#     for k, v in list(seg_metrics.items())[:-1]: 
#         writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
#     for i, opt_group in enumerate(optimizer.param_groups):
#         writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], wrt_step)
        #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

    # RETURN LOSS & METRICS
    log = {'loss': total_loss.average,
            **seg_metrics}

    #if self.lr_scheduler is not None: self.lr_scheduler.step()
    return log

def _valid_epoch(model, epoch):
    if val_loader is None:
        logger.warning('Not data loader was passed for the validation step, No validation is performed !')
        return {}
    logger.info('\n###### EVALUATION ######')

    model.eval()
    wrt_mode = 'val'

    _reset_metrics()
    tbar = tqdm(val_loader, ncols=130)
    with torch.no_grad():
        val_visual = []
        for batch_idx, (data, target) in enumerate(tbar):
            #data, target = data.to(self.device), target.to(self.device)
            # LOSS
            output = model(data)
            loss = loss(output, target)
            if isinstance(loss, torch.nn.DataParallel):
                loss = loss.mean()
            total_loss.update(loss.item())

            seg_metrics = eval_metrics(output, target, num_classes)
<<<<<<< HEAD
            _update_seg_metrics(*seg_metrics, total_correct, total_label, total_inter, total_union)
=======
            _update_seg_metrics(*seg_metrics)
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6

            # LIST OF IMAGE TO VIZ (15 images)
            if len(val_visual) < 15:
                target_np = target.data.cpu().numpy()
                output_np = output.data.max(1)[1].cpu().numpy()
                val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

            # PRINT INFO
<<<<<<< HEAD
            pixAcc, mIoU, _ = _get_seg_metrics(total_correct, total_label, total_inter, total_union).values()
=======
            pixAcc, mIoU, _ = _get_seg_metrics().values()
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
            tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                            total_loss.average,
                                            pixAcc, mIoU))

        # WRTING & VISUALIZING THE MASKS
        val_img = []
        palette = train_loader.dataset.palette
        for d, t, o in val_visual:
            d = restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            [d, t, o] = [viz_transform(x) for x in [d, t, o]]
            val_img.extend([d, t, o])
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
#         writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, wrt_step)

#         # METRICS TO TENSORBOARD
#         wrt_step = (epoch) * len(val_loader)
#         writer.add_scalar(f'{wrt_mode}/loss', total_loss.average, wrt_step)
#         seg_metrics = _get_seg_metrics()
#         for k, v in list(seg_metrics.items())[:-1]: 
#             writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)

        log = {
            'val_loss': total_loss.average,
            **seg_metrics
        }

    return log

<<<<<<< HEAD


# %%
def _update_seg_metrics(correct, labeled, inter, union, total_correct, total_label, total_inter, total_union):
=======
def _reset_metrics():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

def _update_seg_metrics(correct, labeled, inter, union):
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union
<<<<<<< HEAD
    return total_correct, total_label, total_inter, total_union


# %%
def _get_seg_metrics(total_correct, total_label, total_inter, total_union):
    num_classes=21
=======

def _get_seg_metrics(total_correct, total_label, total_inter, total_union):
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
    pixAcc = 1.0 *total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3),
<<<<<<< HEAD
        "Class_IoU": dict(zip(range(num_classes), np.array(np.round(IoU, 3))) )
=======
        "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
    }


# %%
<<<<<<< HEAD
if __name__ == '__main__':
    main(args)

=======
<<<<<<< HEAD
=======


>>>>>>> 4b9378a541d42936800aeb02a24990d2ef4d1350
if __name__ == '__main__':
    main(args)


# %%




>>>>>>> bc52ed36a836d5e0305de2b8c07e34703d4d37d6
