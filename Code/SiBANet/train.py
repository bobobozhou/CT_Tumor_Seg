"""
Scale-Invariant Boundary Aware CNN
by
Bo Zhou,
Carnegie Mellon University,
Merck Sharp & Dohme (MSD),
bzhou2@cs.cmu.edu
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
import transforms_3pair.transforms_3pair as transforms_3pair    # customizd transform for applying same random for 3 images (image, mask, edge)
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
from myloss import *


'''Set up Training Parameters'''
parser = argparse.ArgumentParser(description='Pytorch: Scale-Invariant Boundary Aware CNN')
parser.add_argument('--model_name', default='SiBANet',
                    help='model name used for SiBA-Net')
parser.add_argument('--workers', default=48, type=int, metavar='N',
                    help='number of data loading worker')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of epochs for training network')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=48, type=int, metavar='N',
                    help='mini-batch size for training (default: 64)')
parser.add_argument('--lr', default=0.0002, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_semi', default=0.00001, type=float, metavar='LR',
                    help='semi-training learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for the training optimizer')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--pf', default=1, type=int, metavar='N',
                    help='training print frequency (default: 10)')
parser.add_argument('--df', default=15, type=int, metavar='N',
                    help='training display image frequency (default: 10)')
parser.add_argument('--ef', default=10, type=int, metavar='N',   # 80, 10
                    help='evaluate print frequency (default: 2)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # --resume=models/SiBANet/model_best_SiBANet.pth.tar

'''Set up Training Data Directory'''
parser.add_argument('--train_image_data_dir', default='../../Data_Segmentation/all_train_data_SiBANet/image', type=str, metavar='PATH',
                    help='path to image data')
parser.add_argument('--train_mask_data_dir', default='../../Data_Segmentation/all_train_data_SiBANet/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--train_edge_data_dir', default='../../Data_Segmentation/all_train_data_SiBANet/edge', type=str, metavar='PATH',
                    help='path to edge data')

parser.add_argument('--train_list_filename', default='../../Data_Segmentation/all_train_data_SiBANet/dir/train_list.txt', type=str, metavar='PATH',   ##### need to change
                    help='path to train data list txt file')
parser.add_argument('--semitrain_list_dir', default='../../Data_Segmentation/all_train_data_SiBANet/dir/semi_list/', type=str, metavar='PATH',   ##### need to change
                    help='path to semi-train data list folder, for save or retrieve')

'''Setp up Testing Data Directory'''
# Public TCIA test data
parser.add_argument('--test_image_data_dir_tcia', default='../../Data_Segmentation/public_test_data/image', type=str, metavar='PATH',
                    help='path to image data')
parser.add_argument('--test_mask_data_dir_tcia', default='../../Data_Segmentation/public_test_data/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--test_edge_data_dir_tcia', default='../../Data_Segmentation/public_test_data/edge', type=str, metavar='PATH',
                    help='path to edge data')

parser.add_argument('--test_list_filename_tcia', default='../../Data_Segmentation/public_test_data/dir/test_list_tcia.txt', type=str, metavar='PATH',
                    help='path to tcia test data list txt file')

# MERCK test data
parser.add_argument('--test_image_data_dir_merck', default='../../Data_Segmentation/merck_test_data/image', type=str, metavar='PATH',
                    help='path to image data')
parser.add_argument('--test_mask_data_dir_merck', default='../../Data_Segmentation/merck_test_data/mask', type=str, metavar='PATH',
                    help='path to mask data')
parser.add_argument('--test_edge_data_dir_merck', default='../../Data_Segmentation/merck_test_data/edge', type=str, metavar='PATH',
                    help='path to edge data')

parser.add_argument('--test_list_filename_merck', default='../../Data_Segmentation/merck_test_data/dir/test_list_merck.txt', type=str, metavar='PATH',
                    help='path to merck test data list txt file')

n_classes = 4
class_names = ['Lung', 'Breast', 'Skin', 'Liver']
w_ba = 1; w_rg = 1; w_fin = 1
best_m = 0

epochs_semi_set = [20, 20, 10, 20, 20, 20, 20, 20, 20]  # number of epoch for progressive semi-training
include_semi_set = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]  # iteratively include the slice with distance=1~8 into training


def main():
    global args, best_m
    args = parser.parse_args()

    ''' Initialize and load model (models: VGG16) '''
    model = SiBA_net(fix_para=False)
    print(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True


    ''' Define loss function (criterion) and optimizer '''
    criterion = SoftDiceLoss().cuda()
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
            best_m = checkpoint['best_m']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # 1) Validation data: 
    # (a) Public TCIA Data ~ {144 cases}
    val_dataset_tcia = CTTumorDataset(image_data_dir=args.test_image_data_dir_tcia,
                                      mask_data_dir=args.test_mask_data_dir_tcia,
                                      edge_data_dir=args.test_edge_data_dir_tcia,
                                      list_file=args.test_list_filename_tcia,
                                      transform=transforms_3pair.Compose([transforms_3pair.Resize(112),
                                                                          transforms_3pair.RandomCrop(112),
                                                                          transforms_3pair.ToTensor()]),
                                      norm=transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                      )
    val_loader_tcia = DataLoader(dataset=val_dataset_tcia, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers, pin_memory=True)

    # (b) Merck Data ~ {250 cases} 
    val_dataset_merck = CTTumorDataset(image_data_dir=args.test_image_data_dir_merck,
                                       mask_data_dir=args.test_mask_data_dir_merck,
                                       edge_data_dir=args.test_edge_data_dir_merck,
                                       list_file=args.test_list_filename_merck,
                                       transform=transforms_3pair.Compose([transforms_3pair.Resize(112),
                                                                           transforms_3pair.RandomCrop(112),
                                                                           transforms_3pair.ToTensor()]),
                                       norm=transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       )
    val_loader_merck = DataLoader(dataset=val_dataset_merck, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.workers, pin_memory=True)


    ''' Create logger for recording the training (Tensorboard)'''
    data_logger = Logger('./logs/', name=args.model_name)


    ''' Final Model Validation on Datasets'''
    TEST_STATUS=True
    if TEST_STATUS is True:
        # validate and save TCIA test data
        m_tcia, input_all, output_all, mask_all, case_ind_all = validate(val_loader_tcia, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
        save_seg_montage(input_all, output_all, mask_all, case_ind_all, savefolder='./_RESULTS_montage/tcia/') 
        # save_seg_npy(input_all, output_all, mask_all, case_ind_all, savefolder='./_RESULTS_npy/tcia/') 

        # validate and save MERCK test data
        m_merck, input_all, output_all, mask_all, case_ind_all = validate(val_loader_merck, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
        save_seg_montage(input_all, output_all, mask_all, case_ind_all, savefolder='./_RESULTS_montage/merck/')
        # save_seg_npy(input_all, output_all, mask_all, case_ind_all, savefolder='./_RESULTS_npy/merck/') 


    ############################################################################################################################################
    '''
    STAGE 1: Initial training using all the mid-slice
    '''
    ############################################################################################################################################
    # Generate the training list .txt : initial with only mid-slice = dis_index equal to 0
    semitrain_list_filename = create_semitrainlist(train_list_filename=args.train_list_filename, semitrain_list_dir=args.semitrain_list_dir, 
                                                   dis_ind_set=[0], n=0)

    # Initial training with mid-slice
    STAGE_init = False
    if STAGE_init is True:
        '''Data loading (CT Tumor Dataset): Training Data'''
        # Training data load
        train_dataset = CTTumorDataset(image_data_dir=args.train_image_data_dir,
                                       mask_data_dir=args.train_mask_data_dir,
                                       edge_data_dir=args.train_edge_data_dir,
                                       list_file=semitrain_list_filename,
                                       transform=transforms_3pair.Compose([transforms_3pair.Resize(120),
                                                                           transforms_3pair.RandomRotation(20),
                                                                           transforms_3pair.RandomCrop(112),
                                                                           transforms_3pair.ToTensor()]), 
                                       norm=transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       )
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=True)


        ''' Training for epochs'''
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, data_logger=data_logger, class_names=class_names)
            
            # evaluate on validation set
            if epoch % args.ef == 0 or epoch == args.epochs:
                # m_tcia, _, _, _, _ = validate(val_loader_tcia, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
                m_merck, _, _, _, _ = validate(val_loader_merck, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
                m = m_merck

                # remember best metric and save checkpoint
                is_best = m > best_m
                best_m = max(m, best_m)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_m': best_m,
                    'optimizer': optimizer.state_dict(),
                }, is_best, model=args.model_name)


    ############################################################################################################################################
    '''
    STAGE 2: Progressive semi-supervise training using distanced CT slices (1~8)
    '''
    ############################################################################################################################################
    STAGE_semi = False
    if STAGE_semi is True:
        for i, dis_include in enumerate(include_semi_set):
            ################################# Predict the neighbourhood slice in distance range (Mask, Edge)####################################
            # data loader for Unlabeled image in certain distance range
            unlabel_dataset = CTTumorDataset_unlabel(image_data_dir=args.train_image_data_dir,
                                                     list_file=args.train_list_filename,
                                                     dis_include=dis_include,
                                                     transform=transforms.Compose([transforms.Resize(112),
                                                                                   transforms.RandomCrop(112),
                                                                                   transforms.ToTensor()]),
                                                     norm=transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                                     )
            unlabel_loader = DataLoader(dataset=unlabel_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.workers, pin_memory=True)

            # predict the unlabel data in certain distance range
            print("...... Predicting unlabeled image in distance set: '{}'".format(dis_include))
            case_ind_unlabel, dis_ind_unlabel, img_name_unlabel, img_raw_unlabel, mask_unlabel, edge_unlabel = predict_unlabel(unlabel_loader, model, 
                                                                                                                               percent_keep=0.5,
                                                                                                                               network='SiBA')


            # 1. save predicted Mask & Edge
            # 2. create/append new data to last x.TXT 
            print("...... Saving the new training set to the new training list")
            semitrain_list_filename = save_unlabel(case_ind_unlabel=case_ind_unlabel, dis_ind_unlabel=dis_ind_unlabel, 
                                                   img_save_dir=args.train_image_data_dir, img_name_unlabel=img_name_unlabel,
                                                   mask_save_dir=args.train_mask_data_dir, mask_unlabel=mask_unlabel, 
                                                   edge_save_dir=args.train_edge_data_dir, edge_unlabel=edge_unlabel,
                                                   last_semitrain_list_filename=semitrain_list_filename,
                                                   n=i)
            print("Done!")

            ################################# Semi-Supervised training on dataset with more data ####################################
            # Semi-Training data load
            train_dataset = CTTumorDataset(image_data_dir=args.train_image_data_dir,
                                           mask_data_dir=args.train_mask_data_dir,
                                           edge_data_dir=args.train_edge_data_dir,
                                           list_file=semitrain_list_filename,
                                           transform=transforms_3pair.Compose([transforms_3pair.Resize(120),
                                                                               transforms_3pair.RandomRotation(20),
                                                                               transforms_3pair.RandomCrop(112),
                                                                               transforms_3pair.ToTensor()]), 
                                           norm=transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[2000, 2000, 2000]),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                           )
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers, pin_memory=True)

            # Re-Start optimizer for original learning rate
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr_semi,
                                        momentum=args.momentum,
                                        weight_decay=args.wd)

            # Training for epochs (include predicted neighbourhood slices)
            epochs_semi = epochs_semi_set[i]
            for epoch in range(epochs_semi):
                adjust_learning_rate(optimizer, epoch)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, data_logger=data_logger, class_names=class_names)

                # evaluate on validation set
                if epoch % args.ef == 0 or epoch == args.epochs:
                    # m_tcia, _, _, _, _ = validate(val_loader_tcia, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
                    m_merck, _, _, _, _ = validate(val_loader_merck, model, criterion, epoch=args.start_epoch, data_logger=data_logger, class_names=class_names)
                    m = m_merck

                    # remember best metric and save checkpoint
                    is_best = m > best_m
                    best_m = max(m, best_m)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model': args.model_name,
                        'state_dict': model.state_dict(),
                        'best_m': best_m,
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
        output_ba_all, output_ba1, output_ba2, output_ba3, output_rg_all, output_rg1, output_rg2, output_rg3, output_fin = model(input_var)

        # 2) compute the current loss: loss_boundary, loss_region, loss_final_region
        loss_ba1 = criterion(output_ba1, edge_var)
        loss_ba2 = criterion(output_ba2, edge_var)
        loss_ba3 = criterion(output_ba3, edge_var)
        loss_ba_all = criterion(output_ba_all, edge_var)
        loss_ba = loss_ba1 + loss_ba2 + loss_ba3 + loss_ba_all

        loss_rg1 = criterion(output_rg1, mask_var)
        loss_rg2 = criterion(output_rg2, mask_var)
        loss_rg3 = criterion(output_rg3, mask_var)
        loss_rg_all = criterion(output_rg_all, mask_var)
        loss_rg = loss_rg1 + loss_rg2 + loss_rg3 + loss_rg_all

        loss_fin = criterion(output_fin, mask_var)

        loss = w_ba * loss_ba + w_rg * loss_rg + w_fin * loss_fin

        # 3) record loss and metrics (DSC_slice)
        losses_ba.update(loss_ba.data[0], input.size(0))
        losses_rg.update(loss_rg.data[0], input.size(0))
        losses_fin.update(loss_fin.data[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

        output_fin_np = output_fin.data.cpu().numpy()[:, 0, :, :]       # display predicted & calculate final region
        mask_np = mask_var.data.cpu().numpy()[:, :, :]
        mDSCs, all_DCS_slice = metric_DSC_slice(output_fin_np, mask_np)
        avg_mDSCs.update(mDSCs[0], input.size(0))

        output_ba_np = output_ba_all.data.cpu().numpy()[:, 0, :, :]   # display predicted boundary
        output_rg_np = output_rg_all.data.cpu().numpy()[:, 0, :, :]   # display predicted intermediate region

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
        if epoch % args.df == 0:
            image_disp = np.repeat(input_var.data.cpu().numpy()[0, 0, :, :][np.newaxis, np.newaxis, :, :], 3, axis=1)
            output_ba_disp = make_tf_disp_slice(output_ba_np[0, :, :], edge_var.data.cpu().numpy()[0, :, :])
            output_rg_disp = make_tf_disp_slice(output_rg_np[0, :, :], mask_var.data.cpu().numpy()[0, :, :])
            output_fin_disp = make_tf_disp_slice(output_fin_np[0, :, :], mask_var.data.cpu().numpy()[0, :, :])
            
            tag_inf = '_epoch:' + str(epoch) + ' _iter:' + str(i)
            data_logger.image_summary(tag='train/' + tag_inf + '-0image', images=image_disp, step=i + len(train_loader) * epoch)
            data_logger.image_summary(tag='train/' + tag_inf + '-1image_boundary', images=output_ba_disp, step=i + len(train_loader) * epoch)
            data_logger.image_summary(tag='train/' + tag_inf + '-2image_INTregion', images=output_rg_disp, step=i + len(train_loader) * epoch)
            data_logger.image_summary(tag='train/' + tag_inf + '-3image_FINregion', images=output_fin_disp, step=i + len(train_loader) * epoch)


def validate(val_loader, model, criterion, epoch, data_logger=None, class_names=None):
    losses_ba = AverageMeter()
    losses_rg = AverageMeter()
    losses_fin = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode and evaluate
    model.eval()
    for i, (case_ind, input, mask, edge, class_vec) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, requires_grad=False).cuda()
        mask_var = torch.autograd.Variable(mask).type(torch.FloatTensor).cuda()
        edge_var = torch.autograd.Variable(edge).type(torch.FloatTensor).cuda()

        # 1) output BOUNDARY, REGION, FINAL_REGION from models
        output_ba_all, output_ba1, output_ba2, output_ba3, output_rg_all, output_rg1, output_rg2, output_rg3, output_fin = model(input_var)

        # 2) compute the current loss on validation: loss_boundary, loss_region, loss_final_region
        loss_ba1 = criterion(output_ba1, edge_var)
        loss_ba2 = criterion(output_ba2, edge_var)
        loss_ba3 = criterion(output_ba3, edge_var)
        loss_ba_all = criterion(output_ba_all, edge_var)
        loss_ba = loss_ba1 + loss_ba2 + loss_ba3 + loss_ba_all

        loss_rg1 = criterion(output_rg1, mask_var)
        loss_rg2 = criterion(output_rg2, mask_var)
        loss_rg3 = criterion(output_rg3, mask_var)
        loss_rg_all = criterion(output_rg_all, mask_var)
        loss_rg = loss_rg1 + loss_rg2 + loss_rg3 + loss_rg_all

        loss_fin = criterion(output_fin, mask_var)

        loss = w_ba * loss_ba + w_rg * loss_rg + w_fin * loss_fin

        # 3) record loss and metrics (DSC_volume)
        losses_ba.update(loss_ba.data[0], input.size(0))
        losses_rg.update(loss_rg.data[0], input.size(0))
        losses_fin.update(loss_fin.data[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

        # 4) store all the output, case_ind, gt on validation
        if i == 0:
            case_ind_all = case_ind.cpu().numpy()
            input_all = input_var.data.cpu().numpy()[:,0,:,:]
            mask_all = mask_var.data.cpu().numpy()[:,:,:]
            output_all = generate_CRF(img=input_var.data.cpu().numpy()[:,0,:,:], 
                                      pred=output_fin.data.cpu().numpy()[:,0,:,:],
                                      iter=10, n_labels=2)

        else:
            case_ind_all = np.concatenate((case_ind_all, case_ind.cpu().numpy()), axis=0)
            input_all = np.concatenate((input_all, input_var.data.cpu().numpy()[:,0,:,:]), axis=0)
            mask_all = np.concatenate((mask_all, mask_var.data.cpu().numpy()[:,:,:]), axis=0)
            output_all = np.concatenate((output_all, generate_CRF(img=input_var.data.cpu().numpy()[:,0,:,:],
                                                                  pred=output_fin.data.cpu().numpy()[:,0,:,:],
                                                                  iter=10, n_labels=2)), axis=0)

    # 5) Calcuate the DSC for each volume & the mean DSC
    mDSCv, all_DCS_volume = metric_DSC_volume(output_all, mask_all, case_ind_all)      # np.savetxt('_RESULTS_npy/merck.csv', all_DCS_volume[0], delimiter=',')
    mVSv, all_VS_volume = metric_VS_volume(output_all, mask_all, case_ind_all)
    mHDv, all_HD_volume = metric_HD_volume(output_all, mask_all, case_ind_all)

    # 6) Record loss, m; Visualize the segmentation results (VALIDATE)
    # Print the loss, losses_ba, loss_rg, loss_fin, metric_DSC_slice, every args.print_frequency during training
    print('Epoch: [{0}]\t'
          'meanDSC {meanDSC}\t'
          'medianDSC {medianDSC}\t'
	      'meanVS {meanVS}\t'
	      'medianVS {medianVS}\t'
	      'meanHD {meanHD}\t'
	      'medianHD {medianHD}\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss_BA {loss_ba.val:.4f} ({loss_ba.avg:.4f})\t'
          'Loss_RG {loss_rg.val:.4f} ({loss_rg.avg:.4f})\t'
          'Loss_Fin {loss_fin.val:.4f} ({loss_fin.avg:.4f})\t'.format(epoch,
                                                                      meanDSC=mDSCv[0],
                                                                      medianDSC=np.median(all_DCS_volume[0]),
                                                                      meanVS=mVSv[0],
                                                                      medianVS=np.median(all_VS_volume[0]),
                                                                      meanHD=mHDv[0],
                                                                      medianHD=np.median(all_HD_volume[0]),
                                                                      loss=losses,
                                                                      loss_ba=losses_ba,
                                                                      loss_rg=losses_rg,
                                                                      loss_fin=losses_fin))

    # Plot the training loss, loss_ba, loss_rg, loss_fin, metric_DSC_slice
    data_logger.scalar_summary(tag='validate/loss', value=losses.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/loss_ba', value=losses_ba.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/loss_rg', value=losses_rg.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/loss_fin', value=losses_fin.avg, step=epoch)
    data_logger.scalar_summary(tag='validate/metric_DSC_volume', value=mDSCv[0], step=epoch)

    # Plot the volume segmentation results - Montage
    dict_disp = make_tf_disp_volume(input_all, output_all, mask_all, case_ind_all)
    
    for _, ind in enumerate(dict_disp):
        data_logger.image_summary(tag='validate/case:' + str(ind), images=dict_disp[ind], step=epoch)

    return mDSCv[0], input_all, output_all, mask_all, case_ind_all


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 1000))
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
