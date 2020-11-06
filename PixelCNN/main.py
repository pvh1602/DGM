import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from PIL import Image
from models import * 
from utils import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='../data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='res',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='mnist', help='Can be either cifar|mnist')
parser.add_argument('-r', '--num_resblock', type=int, 
                    default=1, help='The number of [Conv, ReLU] block')  
parser.add_argument('-b', '--batch_size', type=int, 
                    default=128, help='batch size')   
parser.add_argument('-x', '--max_epochs', type=int,
                    default=10, help='How many epochs to run in total?')    
parser.add_argument('-l', '--lr', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-n', '--num_sample', type=int,
                    default=64, help='How many sample to generate?')  
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-c', '--color_condition', help='Using color condition',
                    default=0, type=int)
parser.add_argument('-ln', '--layer_norm', help='Using layer normalization',
                    default=0, type=int)
parser.add_argument('-ml', '--mixture_logistic', help='Using mixture of logistic loss',
                    default=0, type=int)
parser.add_argument('-sk', '--skipped', help='Using skip connection',
                    default=1, type=int)

args = parser.parse_args()
torch.manual_seed(args.seed)
dataset = args.dataset
color_condition = False
use_layer_norm = False
use_mixture_logistic = False
skipped = False
if args.color_condition == 1:
    color_condition = True
if args.layer_norm == 1:
    use_layer_norm = True
if args.mixture_logistic == 1:
    use_mixture_logistic = True
if args.skipped == 1:
    skipped = True


print("====================Loading data====================")
input_size, train_loader, test_loader = data_loader(args.data_dir, args.dataset, args.batch_size)

if dataset == 'mnist':
    model = PixelCNN(input_size, device, args.num_resblock, n_colors=2, 
                    n_filters=64, color_conditioning=color_condition, 
                    use_layer_norm=use_layer_norm,
                    use_mixture_logistic=use_mixture_logistic,
                    skipped=skipped)
elif dataset == 'celeb':    
    model = PixelCNN(input_size, device, args.num_resblock, n_colors=4,
                    n_filters=120, color_conditioning=color_condition, 
                    use_layer_norm=use_layer_norm,
                    use_mixture_logistic=use_mixture_logistic,
                    skipped=skipped)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=1) 
criterion = nn.BCELoss()
epoch = 0

# ============================================
print("==================Training==================")
test_losses = []
train_losses = []
for epoch in range(args.max_epochs):
    model.train()
    train_loss = []
    start_time = time.time()
    
    for batch_idx, inputs in enumerate(train_loader):
        
        inputs = inputs.to(device)
        # print(inputs[0])
        # print("inputs' size: ",inputs.size())
        # outputs = model(inputs)
        loss = model.loss(inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    avg_train_loss = sum(train_loss)/len(train_loss)
    train_losses.extend(train_loss)
    # # decrease learning rate
    # scheduler.step()

    model.eval()
    test_loss = []
    for batch_idx, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        # outputs = model(inputs)
        loss = model.loss(inputs)
        test_loss.append(loss.item())
    avg_test_loss = sum(test_loss)/len(test_loss)
    test_losses.append(avg_test_loss)
    run_time = time.time() - start_time
    print("Epoch {} / {}, time {:.4f}, train loss {:.4f}, test loss {:.4f}".format(epoch, args.max_epochs,
                                                            run_time, avg_train_loss, avg_test_loss))

    # if epoch % 10 == 0:
    #     name = 'sample_{}/layer_norm_{}/seed_{}_resblock_{}_ccon_{}_ne_{}_skipped_{}/epoch_{}.png'.format(dataset, 
    #                 args.layer_norm, args.seed, args.num_resblock, args.color_condition, args.max_epochs, args.skipped,epoch)
    #     samples = model.sample(args.num_sample)
    #     if dataset == 'mnist':
    #         samples = samples.astype('float32') * 255
    #     elif dataset == 'celeb':
    #         samples = samples.astype('float32') / 3 * 255
    #     bits_per_dim = test_losses[epoch] / np.log(2)
    #     title = 'Samples after {} with {:.4f} bits/dim'.format(epoch, bits_per_dim) 
    #     show_samples(samples, fname='./res/'+name, nrow=10, title=title)

print('========================Sampling========================')
name = 'sample_{}/layer_norm_{}/seed_{}_resblock_{}_ccon_{}_ne_{}_skipped_{}/final.png'.format(dataset, 
            args.layer_norm, args.seed, args.num_resblock, args.color_condition, args.max_epochs, args.skipped)
samples = model.sample(args.num_sample)
if dataset == 'mnist':
    samples = samples.astype('float32') * 255
elif dataset == 'celeb':
    samples = samples.astype('float32') / 3 * 255
# utils.save_image(samples, './res/sample.png')
bits_per_dim = test_losses[-1] / np.log(2)
title = 'Samples at final with {:.4f} bits/dim'.format(bits_per_dim) 
show_samples(samples, fname='./res/'+name, nrow=10, title=title)

training_process_title = '{} Dataset Train Plot'.format(dataset)
name = './res/sample_{}/layer_norm_{}/seed_{}_resblock_{}_ccon_{}_ne_{}_skipped_{}/train_plot.png'.format(dataset, 
            args.layer_norm, args.seed, args.num_resblock, args.color_condition, args.max_epochs, args.skipped)

save_training_plot(train_losses, test_losses, training_process_title, name)


# Generate and save sample seperately. This is used to compute FID score.

# name = './sample/sample_{}/layer_norm_{}/seed_{}_resblock_{}_ccon_{}_ne_{}_skipped_{}'.format(dataset, 
#             args.layer_norm, args.seed, args.num_resblock, args.color_condition, args.max_epochs, args.skipped)
# save_samples(samples, name)
                                                            