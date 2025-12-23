import os
import sys
import argparse
import math
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from loss import *
from utils.builder import *
from utils.plotter import *
from models.MLPHeader import MLPHead
from util import *
from utils.eval import *
# from model.ResNet32 import resnet32
# from model.SevenCNN import CNN
# from data.Clothing1M import *
from utils.ema import EMA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import datetime
from utils.logger import Logger


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def build_logger(params):
    if params.ablation:
        logger_root = f'Ablation/{params.dataset}'
    else:
        logger_root = f'Results/{params.dataset}'
    logger_root = str(params.model) + logger_root
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if params.ablation:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    else:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir

def compute_loss(logits, logits_s, y, pesudo, candidate_count, indices, n_classes, config):
    candidate_label = torch.full(size=(y.size(0), n_classes), fill_value=0)
    candidate_label.scatter_(dim=1, index=torch.unsqueeze(y, dim=1).cpu(), value=1)


    # px = logits.softmax(dim=1)
    # _, pesudo = torch.max(px, dim=-1)
    candidate_label.scatter_(dim=1, index=torch.unsqueeze(pesudo, dim=1).cpu(), value=1)
    # _, pred = logits.topk(config.topk, 1, True, True)
    # for i in range(config.topk):
    #     candidate_count[indices, pred[:, i]] += 1
    _,y_ce = torch.max(candidate_count[indices], dim=-1)
    L_PLL = F.cross_entropy(logits, y_ce.cuda())

    complementary_label = (1-candidate_label).cuda()
    # complementary_label = complementary_label.cuda()
    L_NL = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(logits, dim=1)) * complementary_label, dim=1))

    # outputs_s = net(x_s)
    # logits_s = outputs_s['logits'] if type(outputs) is dict else outputs
    L_CR = F.cross_entropy(logits_s, pesudo)

    loss = config.alpha * L_NL + config.alpha * L_CR + config.beta * L_PLL   
    
    return loss


def robust_train(net, optimizer, trainloader, n_classes, candidate_count, config, train_loss_meter,train_accuracy_meter):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='ROBUST TRAINING')
    for it, sample in enumerate(pbar):
        indices = sample['index']
        x, x_s = sample['data']
        x, x_s = x.to(device), x_s.to(device)
        y = sample['label'].to(device)

        candidate_label = torch.full(size=(y.size(0), n_classes), fill_value=0)
        candidate_label.scatter_(dim=1, index=torch.unsqueeze(y, dim=1).cpu(), value=1)

        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs

        px = logits.softmax(dim=1)
        _, pesudo = torch.max(px, dim=-1)
        candidate_label.scatter_(dim=1, index=torch.unsqueeze(pesudo, dim=1).cpu(), value=1)
        _, pred = logits.topk(config.topk, 1, True, True)
        for i in range(config.topk):
            candidate_count[indices, pred[:, i]] += 1
        _,y_ce = torch.max(candidate_count[indices], dim=-1)
        L_PLL = F.cross_entropy(logits, y_ce.cuda())

        complementary_label = (1-candidate_label)
        complementary_label = complementary_label.cuda()
        L_NL = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(logits, dim=1)) * complementary_label, dim=1))

        outputs_s = net(x_s)
        logits_s = outputs_s['logits'] if type(outputs) is dict else outputs
        L_CR = F.cross_entropy(logits_s, pesudo)

        loss = config.alpha * L_NL + config.alpha * L_CR + config.beta * L_PLL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')



def warmup(net, optimizer, trainloader, candidate_count, train_loss_meter, train_accuracy_meter, config):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='WARMUP TRAINING')
    for it, sample in enumerate(pbar):

        indices = sample['index']
        x, _ = sample['data']
        x = x.to(device)
        y = sample['label'].to(device)

        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs

        _, pred = logits.topk(config.topk, 1, True, True)
        for i in range(config.topk):
            candidate_count[indices, pred[:, i]] += 1
        loss_ce = F.cross_entropy(logits, y)
        penalty = conf_penalty(logits)
        loss = loss_ce + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')




def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, min(11, len(lines) + 1)):
        line = lines[-idx].strip()
        epoch, test_acc = line.split(' | ')[0], line.split(' | ')[3]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f} , std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f} , std: {test_acc_list2.std():.2f}')
        print(
            f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}  ,  {test_acc_list2.mean():.2f}±{test_acc_list2.std():.2f} ')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}


def wrapup_training(result_dir, best_accuracy):
    stats = get_baseline_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f"{result_dir}-bestAcc_{best_accuracy:.4f}-lastAcc_{stats['mean']:.4f}")


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='cosine:20,5e-4,100')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--warmup-lr', type=float, default=0.001)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--params-init', type=str, default='none')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-weights', type=bool, default=False)

    parser.add_argument('--rescale-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)
    parser.add_argument('--database', type=str, default='./dataset')

    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--topk', type=int, default=1)



    args = parser.parse_args()
    print(args)
    return args



if __name__ == '__main__':
    config = parse_args()

    assert config.dataset in ['cifar100nc', 'cifar80no']
    assert config.noise_type in ['symmetric', 'asymmetric']
    config.openset_ratio = 0.0 if config.dataset == 'cifar100nc' else 0.2

    init_seeds(config.seed)
    device = set_device(config.gpu)

    # bulid logger
    logger, result_dir = build_logger(config)
    logger.msg(str(config))

    # create dataloader
    loader_dict = build_loader(config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']

    # create model
    # model = build_model(n_classes, config)

    # create optimizer & lr_plan or lr_scheduler
    # optim = build_optimizer(model, config)
    lr_plan = build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr, decay=config.lr_decay,
                            warmup_gradual=config.warmup_gradual)

    # for training
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()

    candidate_count = torch.zeros(n_samples,n_classes)

    epoch = 0

    if config.resume!=None:
        path = config.resume
        dict_s = torch.load(path, map_location='cpu')
        model.load_state_dict(dict_s)
        model.cuda()
        epoch = int(path.split('/')[-1].split('.')[0])+1

    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])
        input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs:
            warmup(model, optim, input_loader, candidate_count, train_loss_meter, train_accuracy_meter, config)
        else:
            robust_train(model, optim, input_loader, n_classes, candidate_count, config, train_loss_meter, train_accuracy_meter)

        eval_result = evaluate_cls_acc(loader_dict['test_loader'], model, device)
        test_accuracy = eval_result['accuracy']

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # if config.save_weights:
            #     torch.save(model.state_dict(), f'{result_dir}/best.pth')
        if config.save_weights and epoch == config.warmup_epochs -1:
            torch.save(model.state_dict(), f'{result_dir}/{epoch}.pth')
        logger.info(
            f'>> Epoch: {epoch} | loss: {train_loss_meter.avg:.2f} | train acc: {train_accuracy_meter.avg:.2f} | test acc: {test_accuracy:.2f} | best acc: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt', layout='1x3')

        epoch+=1

    wrapup_training(result_dir, best_accuracy)