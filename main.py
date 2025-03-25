'''Train CIFAR10/CIFAR100/MNIST/FashionMNIST/Imagenet.'''
import argparse
import os
import csv
from optimizers import (KFACOptimizer, KFACOptimizer_NonFull, EKFACOptimizer, EKFACOptimizer_NonFull, DNGD, SGD_mod, Adam_mod)
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils.data_utils import set_seed
from utils.network_utils_cifar import get_network_cifar
from utils.network_utils_mnist import get_network_mnist
from utils.network_utils_imagenet import get_network_imagenet
from utils.data_utils import NoiseDataLoader
from timm.loss import LabelSmoothingCrossEntropy

def prepare_csv(log_path, dataset, model, depth, optimizer, noise_rate, damping, experiment_type):
    train_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+str(depth)+'/train'
    test_log_path = log_path+'/'+experiment_type+'/'+dataset+'/'+model+str(depth)+'/test'
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path)
    if experiment_type == 'error' and noise_rate==0.0:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'.csv', 'a+', newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'.csv', 'a+', newline='')
    elif experiment_type == 'damping' and noise_rate==0.0:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'_damping'+str(damping)+'.csv', 'a+', newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'_damping'+str(damping)+'.csv', 'a+', newline='')
    else:
        csv_train = open(train_log_path+'/'+optimizer.lower()+'_noise'+str(noise_rate)+'.csv', 'a+', newline='')
        csv_test = open(test_log_path+'/'+optimizer.lower()+'_noise'+str(noise_rate)+'.csv', 'a+', newline='')
    
    csv_train_writer = csv.writer(csv_train)
    csv_test_writer = csv.writer(csv_test)
    return csv_train, csv_train_writer, csv_test, csv_test_writer

def train():
    global best_acc
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--noise_rate', default=0.0, type=float, help="Noise Rate.")
    parser.add_argument('--noise_mode', default='sym', type=str, help="Noise Mode for sym and asym.")
    parser.add_argument('--num_workers', default=0, type=int, help="Num of workers.")
    parser.add_argument('--data_path', default='./data', type=str, help="Data Path of Standard Datasets. Imagenet: D:/Projects/Python/Datasets/ImageNet")
    parser.add_argument('--outputs_dim', default=10, type=int)
    parser.add_argument('--smoothing', default=0.1, type=float)        # label smoothing
    # model parameters
    parser.add_argument('--network', default='vgg16_bn', type=str)
    parser.add_argument('--depth', default=16, type=int)
    parser.add_argument('--growthRate', default=12, type=int)       
    parser.add_argument('--compressionRate', default=2, type=int)      
    parser.add_argument('--widen_factor', default=1, type=int)      
    parser.add_argument('--dropRate', default=0.1, type=float)
    parser.add_argument('--model_path', default='./pretrain', type=str)
    parser.add_argument('--log_path', default='./logs', action='store_true') 
    parser.add_argument('--experiment_type', default='error', type=str) #error noise damping
    parser.add_argument('--random_seed', default=3407, type=int)
    # pyramid
    parser.add_argument('--alpha', default=48, type=int)
    parser.add_argument('--block_type', default='basic', type=str)
    parser.add_argument('--base_channels', default=16, type=int)
    # optimizer argument
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--milestone', default=None, type=str)              # for MultiStepLR
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)           # for KFAC
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--kl_clip', default=1e-2, type=float)          # for KFAC
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--TCov', default=10, type=int)         # for EKFAC
    parser.add_argument('--TScal', default=10, type=int)        # for EKFAC
    parser.add_argument('--TInv', default=10, type=int)        # for EKFAC
    # other argument
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resume', '-r', action='store_true')         # resume from checkpoint
    parser.add_argument('--prefix', default=None, type=str)     # prefix for checkpoint
    parser.add_argument('--load_path', default='', type=str)
    args = parser.parse_args()

    # set_seed(args.random_seed)

    nc = { 
        'mnist': 10,
        'fashionmnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }
    args.outputs_dim = nc[args.dataset.lower()]
    if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist':
        net = get_network_mnist(args.network)
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
        if args.network.lower() == 'preresnet':
            net = get_network_cifar(args.network,depth=args.depth,num_classes=args.outputs_dim)
        elif args.network.lower() == 'pyramidnet':
            net = get_network_cifar(args.network,depth = args.depth,alpha = 48,input_shape=(1, 3, 32, 32),num_classes = args.outputs_dim,base_channels = 16,block_type = 'bottleneck')
        else:
            net = get_network_cifar(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if args.dataset.lower() == 'imagenet':
        if args.network.lower() == 'pyramidnet':
            net = get_network_imagenet(args.network,depth=args.depth,alpha=48,input_shape=(1, 3, 224, 224),num_classes=args.outputs_dim,base_channels=16,block_type='bottleneck')
        else:
            net = get_network_imagenet(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    net = net.to(args.device)
    noisedataloader = NoiseDataLoader(args.dataset.lower(),args.noise_rate,args.noise_mode,args.batch_size,args.num_workers,args.data_path)
    trainloader, testloader = noisedataloader.get_loader()
    optim_name = args.optimizer.lower()
    if optim_name == 'sgd':
        optimizer = optim.SGD(net.parameters(),lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    elif optim_name == 'adam_mod':
        optimizer = Adam_mod(net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    elif optim_name == 'kfac':
        if args.network == 'preresnet':
            optimizer = KFACOptimizer_NonFull(net,lr=args.learning_rate,momentum=args.momentum,stat_decay=args.stat_decay, damping=args.damping,kl_clip=args.kl_clip,weight_decay=args.weight_decay,TCov=args.TCov,TInv=args.TInv)
        else:
            optimizer = KFACOptimizer(net,lr=args.learning_rate,momentum=args.momentum,stat_decay=args.stat_decay,damping=args.damping,kl_clip=args.kl_clip,weight_decay=args.weight_decay,TCov=args.TCov,TInv=args.TInv)
    elif optim_name == 'ekfac':
        if args.network == 'preresnet':
            optimizer = EKFACOptimizer_NonFull(net,lr=args.learning_rate,momentum=args.momentum,stat_decay=args.stat_decay,damping=args.damping,kl_clip=args.kl_clip,weight_decay=args.weight_decay,TCov=args.TCov,TScal=args.TScal,TInv=args.TInv)
        else:
            optimizer = EKFACOptimizer(net,lr=args.learning_rate,momentum=args.momentum,stat_decay=args.stat_decay,damping=args.damping,kl_clip=args.kl_clip,weight_decay=args.weight_decay,TCov=args.TCov,TScal=args.TScal,TInv=args.TInv)
    elif optim_name == 'dngd':
        optimizer = DNGD(net, lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay,damping=args.damping)
    elif optim_name == 'sgd_mod':
        optimizer = SGD_mod(net.parameters(),lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    if args.milestone is None:
        lr_scheduler = MultiStepLR(optimizer, milestones=[10,30,50,100,int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    else:
        milestone = [int(_) for _ in args.milestone.split(',')]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)
    
    if args.smoothing:
        print("Smoothing...")
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_acc = 0
    time_elapsed = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.load_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))
    if args.experiment_type == 'error':
        model_path = args.model_path+'/'+args.experiment_type+'/'+args.dataset.lower()+'/'+args.network.lower()+str(args.depth)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
    csv_train, csv_train_writer, csv_test, csv_test_writer = prepare_csv(args.log_path, args.dataset.lower(), args.network.lower(), args.depth, optim_name, args.noise_rate, args.damping, args.experiment_type)
    csv_train_writer.writerow([
        'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','Dataset:',args.dataset,'Optimizer:',optim_name,'LearningRate:',
        args.learning_rate,'BatchSize:',args.batch_size,'EpochRange:',args.epoch])
    csv_train_writer.writerow(['Epoch','Train_Loss', 'Train_Accuracy','Train_Time'])
    csv_train.flush()
    csv_test_writer.writerow([
        'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','Dataset:',args.dataset,'Optimizer:',optim_name,'LearningRate:',
        args.learning_rate,'BatchSize:',args.batch_size,'EpochRange:',args.epoch])
    csv_test_writer.writerow(['Epoch', 'Test_Loss','Test_Accuracy','Generalization_Gap'])
    csv_test.flush()

    for epoch in range(start_epoch, args.epoch):
        # Train
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        desc = ('[Train][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
            (optim_name, epoch+1, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
        since = time.time()
        for batch_index, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist': inputs = inputs.view(-1, 784)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.to('cpu').data, dim=1),
                                                  1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            loss.backward()
            optimizer.step()
            time_elapsed += time.time() - since
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            desc = ('[Train][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                    (optim_name, epoch+1, lr_scheduler.get_last_lr()[0], train_loss / (batch_index + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
        csv_train_writer.writerow([epoch+1, 
                                   train_loss / (batch_index + 1), 
                                   100.*correct / total,
                                   round(time_elapsed/60,1)])
        csv_train.flush()
        lr_scheduler.step()
        # Validate 
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        desc = ('[Test][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)'
                % (optim_name, epoch+1, lr_scheduler.get_last_lr()[0], test_loss/(0+1), 0, correct, total))
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in prog_bar:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist': inputs = inputs.view(-1, 784)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                desc = ('[Test][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)'
                        % (optim_name, epoch+1, lr_scheduler.get_last_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                prog_bar.set_description(desc, refresh=True)        
        acc = 100.*correct/total
        csv_test_writer.writerow([epoch+1,
                                  test_loss / (batch_index+1),
                                  acc,
                                  abs((test_loss / (batch_index+1)) - (train_loss))
        ])
        csv_test.flush()
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'loss': test_loss,
                'args': args
            }
        if args.experiment_type == 'error':
            torch.save(state, '%s/%s_%s_%s%s_best.t7' % (model_path,
                                                         args.optimizer,
                                                         args.dataset,
                                                         args.network,
                                                         args.depth))
            best_acc = acc          
    csv_train.close()
    csv_test.close()


def main():
    train()
    return best_acc


if __name__ == '__main__':
    main()



