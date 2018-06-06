"""
Scale-Invariant Boundary Aware CNN
by
Bo Zhou,
Carnegie Mellon University,
Merck Sharp & Dohme (MSD),
bzhou2@cs.andrew.edu
"""

import argparse
import os
import shutil
import time
import sys
import ipdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import roc_auc_score, average_precision_score
from logger import Logger
from data_loader import *
from model import *
from utilizes import *


'''Set up Training Parameters'''
parser = argparse.ArgumentParser(description='Pytorch: Scale-Invariant Boundary Aware CNN')
parser.add_argument('--model_name', default='SiBANet',
                    help='model name used for SiBA-Net (no semi-supervised)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading worker')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of epochs for training network')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size for training (default: 64)')
parser.add_argument('--lr', default=0.002, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for the training optimizer')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('-pf', default=10, type=int, metavar='N',
                    help='training print frequency (default: 10)')
parser.add_argument('--ef', default=2, type=int, metavar='N',
                    help='evaluate print frequency (default: 2)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

'''Set up Data Directory'''
parser.add_argument('--image_data_dir', default='../Data/public_data/image', type=str, metavar='PATH',
                    help='path to image data')
parser.add_argument('--mask_data_dir', default='../Data/public_data/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--edge_data_dir', default='../Data/public_data/edge', type=str, metavar='PATH',
                    help='path to edge data')

parser.add_argument('--train_list_dir', default='../Data/public_data/dir/train_list.txt', type=str, metavar='PATH',
                    help='path to train data list txt file')
parser.add_argument('--val_list_dir', default='../Data/public_data/dir/val_list.txt', type=str, metavar='PATH',
                    help='path to validation data list txt file')
parser.add_argument('--test_list_dir', default='../Data/public_data/dir/test_list.txt', type=str, metavar='PATH',
                    help='path to test data list txt file')

n_classes = 4
class_names = ['Lung', 'Breast', 'Skin', 'Liver']
para_mean = np.array([0.485, 0.456, 0.406])
para_std = np.array([0.229, 0.224, 0.225])
w_ba = 1; w_rg = 1; w_fin = 1
best_m = 0


def main():
    global args, best_m1
    args = parser.parse_args()

    ''' Initialize and load model (models: VGG16) '''
    model = SiBA_net(fix_para=False)
    print(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    ''' Define loss function (criterion) and optimizer '''
    # criterion = SoftDiceLoss().cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    ''' Optional: Resume from a checkpoint '''
    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_m1 = checkpoint['best_m1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    '''
    Data loading (CT Tumor Dataset):
    1) Training Data
    2) Validation Data
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 1) training data
    train_dataset = CTTumorDataset(image_data_dir=args.image_data_dir,
                                   mask_data_dir=args.mask_data_dir,
                                   edge_data_dir=args.edge_data_dir,
                                   list_file=args.train_list_dir,
                                   transform=
                                   transforms.Compose(
                                       [transforms.Resize(128),
                                        transforms.RandomCrop(112),
                                        transforms.ToTensor(),
                                        ]), 
                                   norm=normalize)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)

    # 2) validation data
    val_dataset = CTTumorDataset(image_data_dir=args.image_data_dir,
                                 mask_data_dir=args.mask_data_dir,
                                 edge_data_dir=args.edge_data_dir,
                                 list_file=args.test_list_dir,
                                 transform=
                                 transforms.Compose(
                                     [transforms.Resize(128),
                                      transforms.RandomCrop(112),
                                      transforms.ToTensor(),
                                      normalize,
                                      ]), 
                                 norm=normalize)
    val_loader = DataLoader(dataset=val_dataset, batch_size=100000000,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    ''' Create logger for recording the training (Tensorboard)'''
    data_logger = Logger('./logs/', name=args.model_name)

    ''' Training for epochs'''
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, data_logger=data_logger, class_names=class_names)

        # evaluate on validation set
        if epoch % args.ef == 0 or epoch == args.epochs:
            m = validate(val_loader, model, criterion, epoch, data_logger=data_logger, class_names=class_names)
            # remember best prec@1 and save checkpoint
            is_best = m > best_m
            best_m1 = max(m1, best_m1)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_m1': best_m1,
                'optimizer': optimizer.state_dict(),
            }, is_best, model=args.model_name)


def train(train_loader, model, criterion, optimizer, epoch, data_logger=None, class_names=None):
    losses_ba = AverageMeter()
    losses_rg = AverageMeter()
    losses_fin = AverageMeter()
    losses = AverageMeter()
    avg_mDSCs = AverageMeter()

    # switch to training mode and train
    model.train()
    for i, (case_ind, input, mask, edge, class_vec) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input, requires_grad=True).cuda()
        mask_var = torch.autograd.Variable(mask).type(torch.FloatTensor).cuda()
        edge_var = torch.autograd.Variable(edge).type(torch.FloatTensor).cuda()
        # class_vec = class_vec.type(torch.FloatTensor).cuda()
        # class_vec_var = torch.autograd.Variable(class_vec)

        # 1) output BOUNDARY, REGION, FINAL_REGION from models
        output_ba, output_rg, output_fin = model(input_var)

        # 2) compute the current loss: loss_boundary, loss_region, loss_final_region
        loss_ba = criterion(output_ba, edge_var)
        loss_rg = criterion(output_rg, mask_var)
        loss_fin = criterion(output_fin, mask_var)
        loss = w_ba * loss_ba + w_rg * loss_rg + w_fin * loss_fin

        # 3) record loss and metrics (DSC_slice)
        losses_ba.update(loss_ba.data[0], input.size(0))
        losses_rg.update(loss_rg.data[0], input.size(0))
        losses_fin.update(loss_fin.data[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

        output_fin_np = output_fin.data.cpu().numpy()   #display predicted & calculate final region
        mask_np = mask_var.data.cpu().numpy()
        mDSCs, all_DCS_slice = metric_DSC_slice(output_fin_np, mask_np)
        avg_mDSCs.update(mDSCs[0], input.size(0))

        output_ba_np = output_ba.data.cpu().numpy()  #display predicted boundary
        output_rg_np = output_rg.data.cpu().numpy()  #display predicted intermedicate region

        # 5) compute gradient and do SGD step for optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6) Record loss, m; Visualize the segmentation results (TRAINING)
        # Print the loss, losses_ba, loss_rg, loss_fin, metric_DSC_slice, every args.print_frequency during training
        if i % args.pf == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_BA {loss_ba.val:.4f} ({loss_ba.avg:.4f})\t'
                  'Loss_RG {loss_rg.val:.4f} ({loss_rg.avg:.4f})\t'
                  'Loss_Fin {loss_fin.val:.4f} ({loss_fin.avg:.4f})\t'
                  'Metric_DSC_slice {avg_mDSCs.val:.3f} ({avg_mDSCs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                      loss=losses,
                                                                                      loss_ba=losses_ba,
                                                                                      loss_rg=losses_rg,
                                                                                      loss_fin=losses_fin,
                                                                                      avg_mDSCs=avg_mDSCs))

        # Plot the training loss, loss_ba, loss_rg, loss_fin, metric_DSC_slice
        data_logger.scalar_summary(tag='train/loss', value=loss, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/loss_ba', value=loss_ba, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/loss_rg', value=loss_rg, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/loss_fin', value=loss_fin, step=i + len(train_loader) * epoch)
        data_logger.scalar_summary(tag='train/metric_DSC_slice', value=mDSCs[0], step=i + len(train_loader) * epoch)

        # Plot the image segmentation results
        output_ba_disp = make_tf_disp(output_ba_np[0, 0, :, :], edge_var.data.cpu().numpy()[0, 0, :, :])
        output_rg_disp = make_tf_disp(output_rg_np[0, 0, :, :], mask_var.data.cpu().numpy()[0, 0, :, :])
        output_fin_disp = make_tf_disp(output_fin_np[0, 0, :, :], mask_var.data.cpu().numpy()[0, 0, :, :])
        
        tag_inf = '_epoch:' + str(epoch) + ' _iter:' + str(i)
        data_logger.image_summary(tag='train/image_boundary' + tag_inf, images=output_ba_disp, step=i + len(train_loader) * epoch)
        data_logger.image_summary(tag='train/image_INTregion' + tag_inf, images=output_rg_disp, step=i + len(train_loader) * epoch)
        data_logger.image_summary(tag='train/image_FINregion' + tag_inf, images=output_fin_disp, step=i + len(train_loader) * epoch)

        ipdb.set_trace()


def validate(val_loader, model, criterion, epoch, data_logger=None, class_names=None):
    losses = AverageMeter()
    avg_m1 = AverageMeter()

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluation mode and evaluate
    model.eval()
    for i, (case_ind, input, mask, edge, class_vec) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, requires_grad=True).cuda()
        target = target.type(torch.FloatTensor).cuda(async=True)
        target_var = torch.autograd.Variable(target)

        gt = torch.cat((gt, target), 0)  # store/concatenate for overall calculation

        # 1) output score map from models (AlexNet, VGG19, ResNet101, DenseNet169)
        output_scoremap = model(input_var)

        # 2) output maximal score from the score map
        find_max = nn.MaxPool2d(kernel_size=output_scoremap.size()[2:])
        output_max = find_max(output_scoremap)
        output_max = output_max.view(output_max.shape[0], output_max.shape[1])

        pred = torch.cat((pred, output_max.data), 0)  # store/concatenate for overall calculation

        # 3) compute the current loss
        loss = criterion(output_max, target_var)

    # 4) record loss and metrics (mAp & ROC)
    m1, all_ap = metric_DSC_volume(pred, gt)
    avg_m1.update(m1[0], input.size(0))

    # Plot the validation m1, validation m2 curves
    data_logger.scalar_summary(tag='validation/metric1:mAp', value=avg_m1.avg, step=epoch)
    for ii in range(n_classes):
        data_logger.scalar_summary(tag='validation/AUCROC:' + class_names[ii], value=all_aucroc[0][ii], step=epoch)

    print(' * Metric1 {avg_m1.avg:.3f}'.format(avg_m1=avg_m1))
    return avg_m1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, model=None):
    """Save checkpoint and the current best model"""
    save_path = './models/' + str(model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    filename_ckpt = save_path + '/checkpoint_' + str(model) + '.pth.tar'
    filename_best = save_path + '/model_best_' + str(model) + '.pth.tar'

    torch.save(state, filename_ckpt)
    if is_best:
        shutil.copyfile(filename_ckpt, filename_best)

if __name__ == '__main__':
    main()
