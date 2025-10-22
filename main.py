'''Train CIFAR10/CIFAR100/MNIST/FashionMNIST/Imagenet/MiniImagenet/TinyImagenet.'''
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from tqdm import tqdm
from utils.get_networks import get_network
from utils.get_optimizers import get_optimizer
from utils.data_utils import NoiseDataLoader
from utils.log_utils import prepare_csv, write_csv
from timm.loss import LabelSmoothingCrossEntropy

def train():
    global best_acc
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--dataset',default='cifar10',type=str,help="cifar10,cifar100,mnist,fashionmnist,imagenet,miniimagenet.")
    parser.add_argument('--noise_rate',default=0.0,type=float,help="Noise Rate.")
    parser.add_argument('--noise_mode',default='sym',type=str,help="Noise Mode for sym and asym.")
    parser.add_argument('--num_workers',default=8,type=int,help="Num of workers.")
    parser.add_argument('--data_path',default='/nfsshare/home/wupingfan/datasets',type=str,help="Data Path of Standard Datasets.")   # E:\datasets
    parser.add_argument('--outputs_dim',default=10,type=int)
    parser.add_argument('--input_dim',default=32,type=int)
    parser.add_argument('--in_channels',default=3,type=int)
    parser.add_argument('--mlp_ratio',default=4.0,type=float)
    parser.add_argument('--patch_size',default=16,type=int)
    parser.add_argument('--smoothing',default=0.1,type=float)        # label smoothing
    # model parameters
    parser.add_argument('--network',default='vgg16_bn',type=str)
    parser.add_argument('--depth',default=16,type=int)
    parser.add_argument('--growthRate',default=12,type=int)       
    parser.add_argument('--compressionRate',default=2,type=int)      
    parser.add_argument('--widen_factor',default=1,type=int)      
    parser.add_argument('--dropRate',default=0.1,type=float)
    parser.add_argument('--cardinality',default=4,type=int)
    parser.add_argument('--model_path',default='./pretrain',type=str)
    parser.add_argument('--log_path',default='./logs',action='store_true') 
    parser.add_argument('--experiment_type',default='error',type=str, help="error, noise, damping") #error noise damping
    parser.add_argument('--random_seed',default=3407,type=int)
    # pyramid
    parser.add_argument('--alpha',default=48,type=int)
    parser.add_argument('--block_type',default='basic',type=str,help="basic, bottle_neck")
    parser.add_argument('--base_channels',default=16,type=int)
    # optimizer argument
    parser.add_argument('--optimizer',default='adam_',type=str)
    parser.add_argument('--learning_rate',default=0.1,type=float)
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--epoch',default=300,type=int)
    parser.add_argument('--eps',default=1e-10,type=float)             # for Adam AdamW and AdaGrad
    parser.add_argument('--retraction_eps',default=1e-10,type=float)  # for Muon
    parser.add_argument('--amsgrad',default=False,type=bool)    # for Adam 
    parser.add_argument('--betas',default=[0.9,0.999],type=list)    # for Adam, AdamW and Muon
    parser.add_argument('--milestone',default=None,type=str)              # for MultiStepLR
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--stat_decay',default=0.95,type=float)           # for KFAC
    parser.add_argument('--damping',default=1e-3,type=float)
    parser.add_argument('--kl_clip',default=1e-2,type=float)          # for KFAC
    parser.add_argument('--weight_decay',default=5e-4,type=float)
    parser.add_argument('--TCov',default=100,type=int)         # for EKFAC
    parser.add_argument('--TScal',default=100,type=int)        # for EKFAC
    parser.add_argument('--TInv',default=100,type=int)        # for EKFAC

    # other argument
    parser.add_argument('--device',default='cuda',type=str)
    parser.add_argument('--resume','-r',action='store_true')         # resume from checkpoint
    parser.add_argument('--prefix',default=None,type=str)     # prefix for checkpoint
    parser.add_argument('--load_path',default='',type=str)
    args = parser.parse_args()

    noisedataloader = NoiseDataLoader(args.dataset.lower(),args.noise_rate,args.noise_mode,args.batch_size,args.num_workers,args.data_path)
    trainloader,testloader = noisedataloader.get_loader()
    net = get_network(args)
    optimizer,optim_name = get_optimizer(args,net)
    if args.milestone is None:
        if args.network.lower() in ['vit_small','vit_base','vit_large','vit_huge']:
            # lr_scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)
            lr_scheduler = MultiStepLR(optimizer,milestones=[10,50,100,int(args.epoch*0.5),int(args.epoch*0.75)],gamma=0.1)
        else:
            lr_scheduler = CosineAnnealingLR(optimizer,T_max=args.epoch)
            # lr_scheduler = MultiStepLR(optimizer,milestones=[10,30,50,100,int(args.epoch*0.5),int(args.epoch*0.75)],gamma=0.1)
    else:
        milestone = [int(_) for _ in args.milestone.split(',')]
        lr_scheduler = MultiStepLR(optimizer,milestones=milestone,gamma=0.1)
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
        assert os.path.isfile(args.load_path),'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.load_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Loaded checkpoint at epoch: %d,acc: %.2f%%' % (start_epoch,best_acc))
    if args.experiment_type == 'error':
        model_path = args.model_path+'/'+args.experiment_type+'/'+args.dataset.lower()+'/'+args.network.lower()+str(args.depth)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
    csv_train,csv_train_writer,csv_test,csv_test_writer = prepare_csv(args.log_path,args.dataset.lower(),args.network.lower(),args.depth,optim_name,args.noise_rate,args.damping,args.experiment_type)
    write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=True,train=False,tets=False,args=args,optim_name=optim_name)
    for epoch in range(start_epoch,args.epoch):
        # Train
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        desc = ('[Train][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
            (optim_name,epoch+1,lr_scheduler.get_last_lr()[0],0,0,correct,total))
        prog_bar = tqdm(enumerate(trainloader),total=len(trainloader),desc=desc,leave=True)
        since = time.time()
        for batch_index, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            if args.dataset.lower() in ['mnist', 'fashionmnist']:
                inputs = inputs.view(-1, 784)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if optim_name in ['kfac','ekfac'] and optimizer.steps % optimizer.TCov == 0:
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.to('cpu').data, dim=1),
                                                  1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_elapsed += time.time() - since
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            desc = ('[Train][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                    (optim_name, epoch+1, lr_scheduler.get_last_lr()[0],
                     train_loss / (batch_index + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
            prog_bar.update(1)
        prog_bar.close()
        write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=False,train=True,test=False,args=None,
                    epoch=epoch,train_loss=train_loss,correct=correct,total=total,time_elapsed=time_elapsed,batch_index=batch_index)
        lr_scheduler.step()
        # Validate 
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        desc = ('[Test][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)'
                % (optim_name,epoch+1,lr_scheduler.get_last_lr()[0],test_loss/(0+1),0,correct,total))
        prog_bar = tqdm(enumerate(testloader),total=len(testloader),desc=desc,leave=True)
        with torch.no_grad():
            for batch_idx,(inputs,targets) in prog_bar:
                inputs,targets = inputs.to(args.device),targets.to(args.device)
                if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist': inputs = inputs.view(-1,784)
                outputs = net(inputs)
                loss = criterion(outputs,targets)
                test_loss += loss.item()
                _,predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                desc = ('[Test][%s][%s][LR=%.4f] Loss: %.4f | Acc: %.3f%% (%d/%d)'
                        % (optim_name,epoch+1,lr_scheduler.get_last_lr()[0],test_loss / (batch_idx + 1),100. * correct / total,correct,total))
                prog_bar.set_description(desc,refresh=True)        
        acc = 100.*correct/total
        write_csv(csv_train,csv_train_writer,csv_test,csv_test_writer,head=False,train=False,test=True,args=None,
                    epoch=epoch,test_loss=test_loss,acc=acc,batch_index=batch_idx,train_loss=train_loss)
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
            torch.save(state,'%s/%s_%s_%s%s_best.t7' % (model_path,
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



