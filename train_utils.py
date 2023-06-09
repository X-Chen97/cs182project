import argparse
import os
import shutil
import sys
import time
import warnings
warnings.filterwarnings('ignore')
from random import sample
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from data_utils import CIFData
from data_utils import collate_pool, get_train_val_test_loader
# from model_hw import CrystalGraphConvNet


def train(train_loader, model, criterion, optimizer, epoch, normalizer, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Should we include CUDA training?

        input_var = (Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3])
        # normalize target
        target_normed = normalizer.norm(target)

        target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure metrics and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #             'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, mae_errors=mae_errors)
        #     )
    # print the average loss and MAE for the epoch 
    print('Train: \t'
            'Time {batch_time.avg:.3f}\t'
            'Data {data_time.avg:.3f}\t'
            'Loss {loss.avg:.4f}\t'
            'MAE {mae_errors.avg:.3f}'.format(
        batch_time=batch_time,
        data_time=data_time, loss=losses, mae_errors=mae_errors)
    )
    return float(losses.avg), float(mae_errors.avg)

def validate(val_loader, model, criterion, normalizer, test=False, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            input_var = (Variable(input[0]),
                            Variable(input[1]),
                            input[2],
                            input[3])

        target_normed = normalizer.norm(target)
        
        with torch.no_grad():
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure metrics and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #             'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
        #         i, len(val_loader), batch_time=batch_time, loss=losses,
        #         mae_errors=mae_errors))
    # print the average loss and MAE for the epoch
    print('Test: \t'
            'Time {batch_time.avg:.3f}\t'
            'Loss {loss.avg:.4f}\t'
            'MAE {mae_errors.avg:.3f}'.format(
        batch_time=batch_time, loss=losses,
        mae_errors=mae_errors))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    # else:
    #     star_label = '*'

    # print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
    #                                                 mae_errors=mae_errors))
    return float(losses.avg), float(mae_errors.avg)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, lr, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
