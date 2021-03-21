import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
from torchvision import models
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from resnet import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--lambda_par', type=float, default=1.0, help='unlabeled dataset ratio')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
def plant_village_loader():
    train_fold ='/home/yel004/ECE269_final_proj/PlantVillage_Training_Set'
    valid_fold = '/home/yel004/ECE269_final_proj/PlantVillage_Validation_Set'
#     transform = transforms.Compose(
#     [
#         transforms.RandomAffine(degrees=90,translate=(0.1,0.1)),
#         transforms.ToTensor()
#     ])
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_dataset = dset.ImageFolder(root=train_fold, transform=train_transform)
    valid_dataset = dset.ImageFolder(root=valid_fold, transform=valid_transform)

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     validation_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset,valid_dataset

# CIFAR_CLASSES = 10
PLANT_CLASSES = 38
torch.cuda.empty_cache()
def main():
    
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_stud = nn.CrossEntropyLoss()
    criterion_stud = criterion_stud.cuda()
    model = Network(args.init_channels, PLANT_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    student = ResNet(criterion_stud)
#     student = models.resnet18(pretrained = False)
    student = student.cuda()

    optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer_stud = torch.optim.SGD(student.parameters(),args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)

#     train_transform, valid_transform = utils._data_transforms_cifar10(args)
#     train_transform1, valid_transform1 = utils._data_transforms_cifar100(args)
#     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
#     unlabeled = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform1)
    train_data,unlabeled = plant_village_loader()

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=2)

    unlabeled_queue = torch.utils.data.DataLoader(unlabeled, batch_size=args.batch_size, pin_memory=True, num_workers=2, shuffle = True)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, student, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, unlabeled_queue, model, student, architect, criterion, criterion_stud, optimizer, optimizer_stud, lr)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
        
def cusloss(inp, tar):
    m = nn.Softmax(1)
    lm = nn.LogSoftmax(1)
    lenn = inp.shape[0]
    inp = lm(inp)
    tar = m(tar)
    out = inp*tar
    ll = (out.sum()*(-1))/lenn
    return ll

def train(train_queue, valid_queue, unlabeled_queue, model, student, architect, criterion, criterion_stud, optimizer, optimizer_stud, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()


  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()
    
    # get a random minibatch from the cifar-100 queue with replacement
    input_unlabeled, target_unlabeled = next(iter(unlabeled_queue))
    input_unlabeled = Variable(input_unlabeled, requires_grad=False).cuda()
    target_unlabeled = Variable(target_unlabeled, requires_grad=False).cuda()

    architect.step(input, target, input_search, target_search, input_unlabeled, lr, optimizer, unrolled=args.unrolled)
    architect.step1(input, target, input_search, target_search, input_unlabeled, lr, optimizer, optimizer_stud, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()
    
    optimizer_stud.zero_grad()
    l1 = model(input_unlabeled)
    logits1 = student(input_unlabeled)
    loss1 = cusloss(logits1, l1.detach())
    

    loss1.backward()
    #nn.utils.clip_grad_norm(student.parameters(), args.grad_clip)
    optimizer_stud.step()
    
    optimizer_stud.zero_grad()
    logits2 = student(input)
    loss2 = criterion_stud(logits2, target)

    loss2.backward()
    #nn.utils.clip_grad_norm(student.parameters(), args.grad_clip)
    optimizer_stud.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)


  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    

  return top1.avg, objs.avg

if __name__ == '__main__':
  main() 
