import argparse
import ast
import copy
import json
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
import torchvision
from utils.data_utils import NoiseDataLoader
from utils.get_networks import get_network
from utils.get_optimizers import get_optimizer
from utils.log_utils import (
    prepare_csv,
    prepare_sweep_csv,
    prepare_wallclock_csv,
    write_csv,
    write_sweep_result,
    write_wallclock_csv,
)
from utils.sweep_utils import (
    apply_sweep_combo,
    expand_sweep_grid,
    format_sweep_tag,
    parse_sweep_params,
    validate_sweep_params,
)

torchvision.disable_beta_transforms_warning()
TEXT_DATASETS = {'sst2'}
VIT_NETWORKS = {'vit_small', 'vit_base', 'vit_large', 'vit_huge'}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot interpret boolean value: {value}')


def normalize_betas(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
            raise ValueError('--betas must parse to a pair like (0.9, 0.999).')
        return tuple(float(v) for v in parsed)
    raise ValueError(f'Unsupported betas value: {value}')


def normalize_args(args):
    args.dataset = args.dataset.lower()
    args.network = args.network.lower()
    args.optimizer = args.optimizer.lower()
    args.sam_base_optimizer = args.sam_base_optimizer.lower()
    args.betas = normalize_betas(args.betas)
    return args


def build_parser(): 
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='cifar10,cifar100,mnist,fashionmnist,svhn,imagenet,miniimagenet,tinyimagenet,sst2')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='Noise Rate.')
    parser.add_argument('--noise_mode', default='sym', type=str, help='Noise Mode for sym and asym.')
    parser.add_argument('--num_workers', default=0, type=int, help='Num of workers.')
    parser.add_argument('--data_path', default=r'E:\datasets', type=str, help='Data path of datasets.')
    parser.add_argument('--outputs_dim', default=10, type=int)
    parser.add_argument('--input_dim', default=32, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--text_model_name', default='distilbert/distilbert-base-uncased', type=str)

    # model parameters
    parser.add_argument('--network', default='vgg16_bn', type=str)
    parser.add_argument('--depth', default=16, type=int)
    parser.add_argument('--growthRate', default=12, type=int)
    parser.add_argument('--compressionRate', default=2, type=int)
    parser.add_argument('--widen_factor', default=1, type=int)
    parser.add_argument('--dropRate', default=0.1, type=float)
    parser.add_argument('--cardinality', default=4, type=int)
    parser.add_argument('--model_path', default='./pretrain', type=str)
    parser.add_argument('--log_path', default='./logs', type=str)
    parser.add_argument('--experiment_type', default='error', type=str, help='error, noise, damping')
    parser.add_argument('--random_seed', default=3407, type=int)

    # pyramid
    parser.add_argument('--alpha', default=48, type=int)
    parser.add_argument('--block_type', default='basic', type=str, help='basic, bottle_neck')
    parser.add_argument('--base_channels', default=16, type=int)

    # optimizer arguments
    parser.add_argument('--optimizer', default='adam_', type=str)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--eps', default=1e-10, type=float)
    parser.add_argument('--retraction_eps', default=1e-10, type=float)
    parser.add_argument('--amsgrad', default=False, type=str2bool)
    parser.add_argument('--betas', default=(0.9, 0.999))
    parser.add_argument('--milestone', default=None, type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--kl_clip', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--rho', default=0.05, type=float, help='Neighborhood size for SAM.')
    parser.add_argument('--sam_base_optimizer', default='sgd', type=str,
                        help='Base optimizer used by SAM: sgd, sgd_, adam_, adagrad_, adamw')
    parser.add_argument('--TCov', default=100, type=int)
    parser.add_argument('--TScal', default=100, type=int)
    parser.add_argument('--TInv', default=100, type=int)

    # sweep arguments
    parser.add_argument('--run_sweep', action='store_true', help='Run a hyper-parameter sweep instead of a single training run.')
    parser.add_argument('--sweep_optimizer', default=None, type=str,
                        help='Optimizer to sweep. If omitted, uses --optimizer.')
    parser.add_argument('--sweep_params', default='', type=str,
                        help='JSON/Python dict defining the hyper-parameter grid, e.g. "{\"learning_rate\":[1e-3,1e-4],\"weight_decay\":[0.0,0.01]}"')

    # other arguments
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('--load_path', default='', type=str)
    return parser


def prepare_inputs_targets(batch, args):
    inputs, targets = batch
    if isinstance(inputs, dict):
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
    else:
        inputs = inputs.to(args.device)
        if args.dataset in {'mnist', 'fashionmnist'}:
            inputs = inputs.view(-1, 784)
    targets = targets.to(args.device)
    return inputs, targets


def forward_logits(model, inputs):
    outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
    if hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


def build_scheduler(args, optimizer):
    if args.milestone is None:
        if args.network in VIT_NETWORKS:
            milestones = [10, 50, 100, int(args.epoch * 0.5), int(args.epoch * 0.75)]
            milestones = sorted({m for m in milestones if 0 < m < args.epoch})
            return MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return CosineAnnealingLR(optimizer, T_max=args.epoch)
    milestone = [int(_) for _ in args.milestone.split(',') if _.strip()]
    return MultiStepLR(optimizer, milestones=milestone, gamma=0.1)


def build_criterion(args):
    if args.smoothing:
        print('Smoothing...')
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return nn.CrossEntropyLoss()


def maybe_prepare_checkpoint_dir(args):
    if args.experiment_type != 'error':
        return None
    if args.network in VIT_NETWORKS or args.dataset in TEXT_DATASETS:
        model_path = os.path.join(args.model_path, args.experiment_type, args.dataset, args.network)
    else:
        model_path = os.path.join(args.model_path, args.experiment_type, args.dataset, f'{args.network}{args.depth}')
    os.makedirs(model_path, exist_ok=True)
    return model_path


def train_one_run(args, extra_tag=None):

    noisedataloader = NoiseDataLoader(
        args.dataset,
        args.noise_rate,
        args.noise_mode,
        args.batch_size,
        args.num_workers,
        args.data_path,
        text_model_name=args.text_model_name,
        max_seq_length=args.max_seq_length,
    )
    trainloader, testloader = noisedataloader.get_loader()

    net = get_network(args)
    optimizer, optim_name = get_optimizer(args, net)
    lr_scheduler = build_scheduler(args, optimizer)
    criterion = build_criterion(args)
    checkpoint_dir = maybe_prepare_checkpoint_dir(args)

    csv_train, csv_train_writer, csv_test, csv_test_writer = prepare_csv(
        args.log_path,
        args.dataset,
        args.network,
        args.depth,
        optim_name,
        args.noise_rate,
        args.damping,
        args.experiment_type,
        extra_tag=extra_tag,
    )
    wallclock_csv, wallclock_writer = prepare_wallclock_csv(
        args.log_path,
        args.dataset,
        args.network,
        args.depth,
        optim_name,
        args.noise_rate,
        args.damping,
        args.experiment_type,
        extra_tag=extra_tag,
    )
    write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer,
              head=True, train=False, test=False, args=args, optim_name=optim_name)
    write_wallclock_csv(wallclock_csv, wallclock_writer, head=True, args=args, optim_name=optim_name)

    start_epoch = 0
    best_acc = 0.0
    best_state = None
    cumulative_wallclock = 0.0
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_train_acc = 0.0
    best_test_acc = 0.0

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.load_path, map_location=args.device)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

    for epoch in range(start_epoch, args.epoch):
        epoch_start = time.time()

        # Train
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        desc = ('[Train][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
        batch_index = -1
        for batch_index, batch in prog_bar:
            inputs, targets = prepare_inputs_targets(batch, args)
            optimizer.zero_grad()
            logits = forward_logits(net, inputs)
            loss = criterion(logits, targets)

            if args.optimizer == 'sam':
                loss.backward()
                optimizer.ascent_step()
                sam_logits = forward_logits(net, inputs)
                sam_loss = criterion(sam_logits, targets)
                sam_loss.backward()
                optimizer.descent_step()
            else:
                if optim_name in {'kfac', 'ekfac'} and optimizer.steps % optimizer.TCov == 0:
                    optimizer.acc_stats = True
                    with torch.no_grad():
                        sampled_y = torch.multinomial(torch.nn.functional.softmax(logits.detach().cpu(), dim=1), 1).squeeze(1).to(args.device)
                    loss_sample = criterion(logits, sampled_y)
                    loss_sample.backward(retain_graph=True)
                    optimizer.acc_stats = False
                    optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            avg_train_loss = train_loss / (batch_index + 1)
            train_acc = 100.0 * correct / total
            desc = ('[Train][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                    (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], avg_train_loss, train_acc, correct, total))
            prog_bar.set_description(desc, refresh=True)
            prog_bar.update(0)
        prog_bar.close()

        avg_train_loss = train_loss / max(batch_index + 1, 1)
        train_acc = 100.0 * correct / max(total, 1)
        best_train_loss = min(best_train_loss, avg_train_loss)
        best_train_acc = max(best_train_acc, train_acc)
        write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer, head=False, train=True, test=False,
                  args=None, epoch=epoch, train_loss=avg_train_loss, correct=correct, total=total)

        lr_scheduler.step()

        # Validate
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        desc = ('[Test][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
        batch_idx = -1
        with torch.no_grad():
            for batch_idx, batch in prog_bar:
                inputs, targets = prepare_inputs_targets(batch, args)
                logits = forward_logits(net, inputs)
                loss = criterion(logits, targets)
                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                avg_test_loss = test_loss / (batch_idx + 1)
                test_acc = 100.0 * correct / total
                desc = ('[Test][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                        (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], avg_test_loss, test_acc, correct, total))
                prog_bar.set_description(desc, refresh=True)
        prog_bar.close()

        avg_test_loss = test_loss / max(batch_idx + 1, 1)
        acc = 100.0 * correct / max(total, 1)
        best_test_loss = min(best_test_loss, avg_test_loss)
        best_test_acc = max(best_test_acc, acc)
        write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer, head=False, train=False, test=True,
                  args=None, epoch=epoch, test_loss=avg_test_loss, acc=acc, train_loss=avg_train_loss)

        epoch_wallclock = time.time() - epoch_start
        cumulative_wallclock += epoch_wallclock
        write_wallclock_csv(
            wallclock_csv,
            wallclock_writer,
            head=False,
            epoch=epoch,
            epoch_wallclock_sec=epoch_wallclock,
            cumulative_wallclock_sec=cumulative_wallclock,
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'loss': avg_test_loss,
                'args': vars(args),
            }
            if checkpoint_dir is not None:
                checkpoint_name = f'{args.optimizer}_{args.dataset}_{args.network}'
                if args.dataset not in TEXT_DATASETS and args.network not in VIT_NETWORKS:
                    checkpoint_name += f'{args.depth}'
                if extra_tag:
                    checkpoint_name += f'_{extra_tag}'
                torch.save(best_state, os.path.join(checkpoint_dir, checkpoint_name + '_best.t7'))

    csv_train.close()
    csv_test.close()
    wallclock_csv.close()

    return {
        'optimizer': optim_name,
        'best_train_loss': best_train_loss,
        'best_test_loss': best_test_loss,
        'best_train_accuracy': best_train_acc,
        'best_test_accuracy': best_test_acc,
        'wallclock_sec': cumulative_wallclock,
        'best_state': best_state,
    }


def run_sweep(args):
    sweep_optimizer = (args.sweep_optimizer or args.optimizer).lower()
    param_grid = parse_sweep_params(args.sweep_params)
    validate_sweep_params(sweep_optimizer, param_grid)
    combos = expand_sweep_grid(param_grid)
    if not combos:
        raise ValueError('No sweep combinations were generated. Check --sweep_params.')

    sweep_csv, sweep_writer = prepare_sweep_csv(
        args.log_path,
        args.dataset,
        args.network,
        args.depth,
        sweep_optimizer,
        args.experiment_type,
    )

    for combo_index, combo in enumerate(combos, start=1):
        run_args = apply_sweep_combo(copy.deepcopy(args), sweep_optimizer, combo)
        run_args.resume = False
        extra_tag = format_sweep_tag(combo_index, combo)
        print(f'Running sweep combo {combo_index}/{len(combos)}: {json.dumps(combo, default=str)}')
        metrics = train_one_run(run_args, extra_tag=extra_tag)
        write_sweep_result(sweep_csv, sweep_writer, combo_index, combo, metrics)

    sweep_csv.close()


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    return normalize_args(args)


def main():
    args = parse_args()
    if args.run_sweep:
        run_sweep(args)
        return None
    metrics = train_one_run(args)
    return metrics['best_test_accuracy']


if __name__ == '__main__':
    main()