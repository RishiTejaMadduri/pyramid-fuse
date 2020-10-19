#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function
import os
import cv2
import tqdm
import json
import argparse
import numpy as np
from PIL import Image
from imageio import imwrite
from torch.utils import data
from torchvision import transforms
import Utils
from models.pyramid_fusion import PyFuse
import utils_seg
import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils_seg import transforms as local_transforms
from base import DataPrefetcher
from utils_seg.helpers import colorize_mask
from utils_seg.metrics import eval_metrics, AverageMeter
from utils_seg.losses import *
from tqdm import tqdm
from dataloaders import *


# In[5]:


parser = argparse.ArgumentParser(description='BiFuse script for 360 Semantic Segmentation!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', default='', type=str, help='Path of source images')
parser.add_argument('--batch_size', default= 16, type=int, help='batch size')
parser.add_argument('--checkpoint_dir', default=None, type=str, help='Path to the saving .pth model')
parser.add_argument('--log_dir', default=None, type=str, help='Path to the saving .pth model')
parser.add_argument('-resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
parser.add_argument('--d', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--val', default = True, type = bool, help = 'Perform validation or not')
parser.add_argument('--epoch', default = 100, type = int, help = 'No of epochs')
parser.add_argument('--momentum', default = 0.99, type = float, help = 'Momentum')
parser.add_argument('--beta', default = 0.01, type = float, help = 'Beta')
parser.add_argument('--decay', default = 10e-5, type = float,help = 'weight_decay')
parser.add_argument('--lr', default = 0.001, type = float, help = 'learning_rate')
parser.add_argument('--val_per_epoch', default = 10, type = int, help = 'validation per epoch')
parser.add_argument('--early_stop', default = 10, type = int, help = 'early stop')
args = parser.parse_args()


# In[4]:


class MyData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        rgb_img = Image.open(img_path).convert("RGB")
        rgb_img = np.array(rgb_img, np.float32) / 255
        rgb_img = cv2.resize(rgb_img, (1024, 512), interpolation=cv2.INTER_AREA)
        data = self.transforms(rgb_img)

        return data

    def __len__(self):
        return len(self.imgs)


# In[11]:


def get_available_devices():
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        self.logger.warning('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu

    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
    available_gpus = list(range(n_gpu))
    return device, available_gpus


# In[1]:


def main(args):
    
#     train_logger = logger()

    train_loader = VOC(args.path, args.batch_size, 'train')
    print(train_loader)
    val_loader = VOC(args.path, args.batch_size, 'val')
    print(val_loader)
#     device, available_gpus = get_available_devices()
    model = PyFuse(50)
    print(f'\n{model}\n')
    model = torch.nn.DataParallel(model, available_gpus)
#     model.to(device)
    loss = CrossEntropyLoss2d()
    print(loss)
    # OPTIMIZER
    optim_params = [
    {'params': model.parameters(), 'lr': args.lr},
    ]

    optimizer = torch.optim.Adam(optim_params,
                             betas=(args.momentum, args.beta),
                             weight_decay=args.weight_decay)
    
#     wrt_mode, wrt_step = 'train_', 0
        
       
        
    # TRANSORMS FOR VISUALIZATION
#     restore_transform = transforms.Compose([
#         local_transforms.DeNormalize(train_loader.MEAN, train_loader.STD),
#         transforms.ToPILImage()])
#     viz_transform = transforms.Compose([
#         transforms.Resize((400, 400)),
#         transforms.ToTensor()])

#     if device ==  torch.device('cpu'): prefetch = False
#     if prefetch:
#         train_loader = DataPrefetcher(train_loader, device=self.device)
#         val_loader = DataPrefetcher(val_loader, device=self.device)

#     torch.backends.cudnn.benchmark = True

        
#     # MONITORING
#     if args.monitor == 'off':
#         mnt_mode = 'off'
#         mnt_best = 0
#     else:
#         mnt_mode, mnt_metric = monitor.split()
#         assert mnt_mode in ['min', 'max']
#         mnt_best = -math.inf if mnt_mode == 'max' else math.inf
#         early_stoping = args.early_stop

#     # CHECKPOINTS & TENSOBOARD
#     start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
#     checkpoint_dir = os.path.join(save_dir, name, start_time)
#     helpers.dir_exists(checkpoint_dir)
#     config_save_path = os.path.join(checkpoint_dir, 'config.json')
#     with open(config_save_path, 'w') as handle:
#         json.dump(self.config, handle, indent=4, sort_keys=True)

#     writer_dir = os.path.join(log_dir, name, start_time)
#     self.writer = tensorboard.SummaryWriter(writer_dir)

#     if resume: _resume_checkpoint(resume)

#     for epoch in range(start_epoch, epochs+1):
#         # RUN TRAIN (AND VAL)
#         results = _train_epoch(epoch)
#         if do_validation and epoch % args.val_per_epoch == 0:
#             results = _valid_epoch(epoch)

#             # LOGGING INFO
#             logger.info(f'\n         ## Info for epoch {epoch} ## ')
#             for k, v in results.items():
#                 logger.info(f'         {str(k):15s}: {v}')

#         if train_logger is not None:
#             log = {'epoch' : epoch, **results}
#             train_logger.add_entry(log)

#         # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
#         if mnt_mode != 'off' and epoch % args.val_per_epoch == 0:
#             try:
#                 if mnt_mode == 'min': improved = (log[mnt_metric] < mnt_best)
#                 else: improved = (log[mnt_metric] > mnt_best)
#             except KeyError:
#                 logger.warning(f'The metrics being tracked ({mnt_metric}) has not been calculated. Training stops.')
#                 break

#             if improved:
#                 mnt_best = log[mnt_metric]
#                 not_improved_count = 0
#             else:
#                 not_improved_count += 1

#             if not_improved_count > early_stoping:
#                 logger.info(f'\nPerformance didn\'t improve for {early_stoping} epochs')
#                 logger.warning('Training Stoped')
#                 break

#         # SAVE CHECKPOINT
#         if epoch % args.save_period == 0:
#             self._save_checkpoint(epoch, save_best=self.improved)


# In[1]:


def _save_checkpoint(self, epoch, save_best=False):
    state = {
        'arch': type(self.model).__name__,
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'monitor_best': self.mnt_best,
        'config': self.config
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
    logger.info(f'\nSaving a checkpoint: {filename} ...') 
    torch.save(state, filename)

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(state, filename)
        self.logger.info("Saving current best: best_model.pth")


# In[2]:


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


# In[7]:


class Trainer():
    def __init__(self, model, loss, resume, train_loader, num_classes, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, train_loader, num_classes, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        
        self.num_classes = num_classes
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = train_logger
        
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            assert output[0].size()[2:] == target.size()[1:]
            assert output[0].size()[1] == self.num_classes 
            loss = self.loss(output[0], target)
            loss += self.loss(output[1], target) * 0.4
            output = output[0]

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            
            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }


# In[ ]:

if __name__=='__main__':
    main(args)



