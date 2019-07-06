# ---------------------------------------------
# 2019 DLP final project
# implementation of Pairwise Body-Part Attention for Recognizing Human-Object Interactions
# Written by Chihchia Li, alter027
# ---------------------------------------------

import argparse
import scipy.io as sio
import numpy as np
import torch
torch.cuda.set_device(2)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import sklearn

import model.dataset as dataset
import model.BPA_model as BPA_model



def parse_args():
	parser = argparse.ArgumentParser(description='Train the Body_Part Attention model')
	parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='HICO', type=str)
    # parser.add_argument('--net', dest='net',
    #                     help='vgg16, res101',
    #                     default='vgg16', type=str)
	parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=200, type=int)

	parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save weights', default="weights",
                        type=str)
	parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=10, type=int)

    # config optimization
	parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-5, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=8, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
	parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
	parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)

	args = parser.parse_args()
	return args

def Loss(y, y_true):
    loss = 10 * y_true * torch.log(y) + 1 * (1-y_true) * torch.log(1-y)
    loss[loss!=loss] = 0
    loss = loss.sum()
    return loss

def compute_mean_avg_prec(y_true, y_score):
    try:
        avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
        mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
    except ValueError:
        mean_avg_prec = 0
    return mean_avg_prec

if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

	## prepare data
    if args.dataset == 'HICO':
        train_dataset = dataset.HICO('train')
        test_dataset = dataset.HICO('test')

    ## prepare models as optimizer
    model = BPA_model.BPA(device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    # for i in range(10):
    #     print(i, optimizer)
    #     lr_scheduler.step()

    ## prepare dataloader
    train_loader = DataLoader(dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2)

    ## start training
    for epoch in range(args.max_epochs):
        print(f'Epoch {epoch}')
        ## training
        accurate, total = 0, 0
        for idx, (x, y, img_names, orig_size) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            output = model(x, img_names, 'train', orig_size)
            loss = loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total += y[y==y].sum()
            accurate += (output==y).sum()

            if idx%args.disp_interval == 0:
                print(f'acc: {accurate/total}')
                accurate, total = 0, 0
        
        accurate, total = 0, 0
        mAP = 0
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            output = model(x, 'test')

            total += y[y==y].sum()
            accurate += (output==y).sum()
            mAP += compute_mean_avg_prec(y, output)
        
        print(f'acc: {accurate/total}, mAP: {mAP/len(test_loader)}')