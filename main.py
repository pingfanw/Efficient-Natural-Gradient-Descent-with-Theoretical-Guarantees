import argparse
import ast
import copy
import json
import os
import re
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from utils.data_utils import NoiseDataLoader
from utils.distributed_utils import (
    cleanup_distributed,
    get_model_state_dict,
    get_world_size,
    is_distributed_training,
    is_main_process,
    maybe_wrap_ddp,
    reduce_metrics,
    setup_distributed,
    unwrap_distributed_model,
)
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
TEXT_DATASETS = {'sst2', 'gsm8k'}
GENERATIVE_TEXT_DATASETS = {'gsm8k'}
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
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='cifar10,cifar100,mnist,fashionmnist,svhn,imagenet,miniimagenet,tinyimagenet,sst2,gsm8k')
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
    parser.add_argument('--eval_batch_size', default=None, type=int,
                        help='Optional eval batch size. Defaults to --batch_size, except GSM8K defaults to 1.')
    parser.add_argument('--gsm8k_config', default='main', type=str, help='Hugging Face GSM8K config: main or socratic.')
    parser.add_argument('--generation_max_new_tokens', default=256, type=int,
                        help='Maximum number of new tokens to generate during GSM8K evaluation.')

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

    parser.add_argument('--alpha', default=48, type=int)
    parser.add_argument('--block_type', default='basic', type=str, help='basic, bottle_neck')
    parser.add_argument('--base_channels', default=16, type=int)

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

    parser.add_argument('--run_sweep', action='store_true', help='Run a hyper-parameter sweep instead of a single training run.')
    parser.add_argument('--sweep_optimizer', default=None, type=str,
                        help='Optimizer to sweep. If omitted, uses --optimizer.')
    parser.add_argument('--sweep_params', default='', type=str,
                        help='JSON/Python dict defining the hyper-parameter grid, e.g. "{\"learning_rate\":[1e-3,1e-4],\"weight_decay\":[0.0,0.01]}"')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--distributed_training', default=False, type=str2bool,
                        help='Enable/disable DDP-based distributed training.')
    parser.add_argument('--gpu_numbers', default=1, type=int,
                        help='Number of GPUs to use when --distributed_training=True.')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('--load_path', default='', type=str)
    return parser


def prepare_batch(batch, args):
    if args.dataset in GENERATIVE_TEXT_DATASETS:
        if not isinstance(batch, dict):
            raise TypeError(f'Expected GSM8K batch to be a dict, but got {type(batch)}.')
        tensor_batch = {}
        metadata = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                tensor_batch[key] = value.to(args.device, non_blocking=True)
            else:
                metadata[key] = value
        return tensor_batch, metadata

    inputs, targets = batch
    if isinstance(inputs, dict):
        inputs = {key: value.to(args.device, non_blocking=True) for key, value in inputs.items()}
    else:
        inputs = inputs.to(args.device, non_blocking=True)
        if args.dataset in {'mnist', 'fashionmnist'}:
            inputs = inputs.view(-1, 784)
    targets = targets.to(args.device, non_blocking=True)
    return inputs, targets


def forward_outputs(model, inputs):
    return model(**inputs) if isinstance(inputs, dict) else model(inputs)


def forward_logits(model, inputs):
    outputs = forward_outputs(model, inputs)
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
    if args.dataset in GENERATIVE_TEXT_DATASETS:
        if is_main_process(args):
            print('Using model-native causal LM loss for generative text training.')
        return None
    if args.smoothing:
        if is_main_process(args):
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


def extract_gsm8k_final_answer(text):
    if text is None:
        return ''
    text = str(text).strip()
    marker_matches = re.findall(r'####\s*([^\n]+)', text)
    if marker_matches:
        candidate = marker_matches[-1].strip()
    else:
        number_matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        candidate = number_matches[-1].strip() if number_matches else text
    return candidate.replace(',', '').strip()




def build_gsm8k_sampled_labels_for_kfac(logits, labels):
    sampled_labels = labels.detach().clone()
    shift_logits = logits.detach()[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels.ne(-100)

    if valid_mask.any():
        probs = F.softmax(shift_logits.float(), dim=-1)
        sampled_next = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(shift_labels)
        sampled_labels[..., 1:] = torch.where(valid_mask, sampled_next, shift_labels.new_full(shift_labels.shape, -100))
    else:
        sampled_labels[..., 1:] = -100

    return sampled_labels
def compute_gsm8k_token_accuracy(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    if valid_mask.sum().item() == 0:
        return 0, 0
    predictions = shift_logits.argmax(dim=-1)
    correct = predictions.eq(shift_labels) & valid_mask
    return int(correct.sum().item()), int(valid_mask.sum().item())


def _generate_with_model(net, generation_inputs, tokenizer, args):
    generate_fn = getattr(net, 'generate', None)
    if callable(generate_fn):
        return generate_fn(
            **generation_inputs,
            max_new_tokens=args.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    inner = unwrap_distributed_model(net)
    if not hasattr(inner, 'generate'):
        raise AttributeError('The current model does not expose a generate() method.')
    return inner.generate(
            **generation_inputs,
            max_new_tokens=args.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


def evaluate_gsm8k_generation(net, batch_inputs, metadata, tokenizer, args):
    generation_inputs = {
        'input_ids': batch_inputs['generation_input_ids'],
        'attention_mask': batch_inputs['generation_attention_mask'],
    }
    generated_ids = _generate_with_model(net, generation_inputs, tokenizer, args)

    prompt_length = generation_inputs['input_ids'].shape[1]
    predictions = []
    for sample_idx in range(generated_ids.size(0)):
        continuation_ids = generated_ids[sample_idx][prompt_length:]
        prediction_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        predictions.append(prediction_text)

    gold_answers = metadata['final_answer']
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    correct = 0
    for prediction_text, gold_answer in zip(predictions, gold_answers):
        predicted_answer = extract_gsm8k_final_answer(prediction_text)
        if predicted_answer == str(gold_answer).strip():
            correct += 1
    return correct, len(gold_answers)


def resolve_local_batch_sizes(args):
    world_size = max(get_world_size(args), 1)
    if not is_distributed_training(args):
        eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
        if args.dataset == 'gsm8k' and args.eval_batch_size is None:
            eval_batch_size = 1
        return args.batch_size, eval_batch_size

    local_batch_size = max(1, (args.batch_size + world_size - 1) // world_size)
    base_eval_batch = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    local_eval_batch_size = max(1, (base_eval_batch + world_size - 1) // world_size)
    if args.dataset == 'gsm8k' and args.eval_batch_size is None:
        local_eval_batch_size = 1

    if is_main_process(args) and args.batch_size % world_size != 0:
        print(
            f'Warning: global batch_size={args.batch_size} is not divisible by gpu_numbers={world_size}. '
            f'Using local batch_size={local_batch_size}, so the effective global batch becomes {local_batch_size * world_size}.'
        )
    return local_batch_size, local_eval_batch_size


def maybe_create_loggers(args, optim_name, extra_tag=None):
    if not is_main_process(args):
        return (None, None, None, None), (None, None)

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
    return (csv_train, csv_train_writer, csv_test, csv_test_writer), (wallclock_csv, wallclock_writer)


def close_loggers(csv_handles, wallclock_handles):
    csv_train, _, csv_test, _ = csv_handles
    wallclock_csv, _ = wallclock_handles
    if csv_train is not None:
        csv_train.close()
    if csv_test is not None:
        csv_test.close()
    if wallclock_csv is not None:
        wallclock_csv.close()


def train_one_run(args, extra_tag=None):
    local_batch_size, local_eval_batch_size = resolve_local_batch_sizes(args)
    noisedataloader = NoiseDataLoader(
        args.dataset,
        args.noise_rate,
        args.noise_mode,
        local_batch_size,
        args.num_workers,
        args.data_path,
        text_model_name=args.text_model_name,
        max_seq_length=args.max_seq_length,
        eval_batch_size=local_eval_batch_size,
        gsm8k_config=args.gsm8k_config,
    )
    trainloader, testloader = noisedataloader.get_loader(
        distributed_training=is_distributed_training(args),
        rank=getattr(args, 'rank', 0),
        world_size=get_world_size(args),
    )

    net = get_network(args)

    start_epoch = 0
    best_acc = 0.0
    best_state = None
    if args.resume:
        if is_main_process(args):
            print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.load_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint.get('acc', 0.0)
        start_epoch = checkpoint.get('epoch', -1) + 1

    net = maybe_wrap_ddp(net, args)
    optimizer, optim_name = get_optimizer(args, net)
    lr_scheduler = build_scheduler(args, optimizer)
    criterion = build_criterion(args)
    checkpoint_dir = maybe_prepare_checkpoint_dir(args) if is_main_process(args) else None
    csv_handles, wallclock_handles = maybe_create_loggers(args, optim_name, extra_tag=extra_tag)
    csv_train, csv_train_writer, csv_test, csv_test_writer = csv_handles
    wallclock_csv, wallclock_writer = wallclock_handles

    cumulative_wallclock = 0.0
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_train_acc = 0.0
    best_test_acc = 0.0

    generation_tokenizer = getattr(trainloader.dataset, 'tokenizer', None) if args.dataset in GENERATIVE_TEXT_DATASETS else None
    if generation_tokenizer is None and args.dataset in GENERATIVE_TEXT_DATASETS:
        generation_tokenizer = getattr(testloader.dataset, 'tokenizer', None)

    for epoch in range(start_epoch, args.epoch):
        epoch_start = time.time()
        if is_distributed_training(args):
            if getattr(trainloader, 'sampler', None) is not None and hasattr(trainloader.sampler, 'set_epoch'):
                trainloader.sampler.set_epoch(epoch)
            if getattr(testloader, 'sampler', None) is not None and hasattr(testloader.sampler, 'set_epoch'):
                testloader.sampler.set_epoch(epoch)

        net.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        num_train_batches = 0
        desc = ('[Train][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
        if is_main_process(args):
            prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
        else:
            prog_bar = enumerate(trainloader)

        batch_index = -1
        for batch_index, batch in prog_bar:
            optimizer.zero_grad()

            if args.dataset in GENERATIVE_TEXT_DATASETS:
                batch_inputs, _ = prepare_batch(batch, args)
                model_inputs = {
                    'input_ids': batch_inputs['input_ids'],
                    'attention_mask': batch_inputs['attention_mask'],
                    'labels': batch_inputs['labels'],
                }
                outputs = forward_outputs(net, model_inputs)
                loss = outputs.loss

                if args.optimizer == 'sam':
                    loss.backward()
                    optimizer.ascent_step()
                    sam_outputs = forward_outputs(net, model_inputs)
                    sam_loss = sam_outputs.loss
                    sam_loss.backward()
                    optimizer.descent_step()
                else:
                    if optim_name in {'kfac', 'ekfac'} and optimizer.steps % optimizer.TCov == 0:
                        optimizer.acc_stats = True
                        sampled_labels = build_gsm8k_sampled_labels_for_kfac(outputs.logits, batch_inputs['labels'])
                        sampled_inputs = {
                            'input_ids': batch_inputs['input_ids'],
                            'attention_mask': batch_inputs['attention_mask'],
                            'labels': sampled_labels,
                        }
                        loss_sample = forward_outputs(net, sampled_inputs).loss
                        loss_sample.backward(retain_graph=True)
                        optimizer.acc_stats = False
                        optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                batch_correct, batch_total = compute_gsm8k_token_accuracy(outputs.logits, batch_inputs['labels'])
                correct += batch_correct
                total += batch_total
            else:
                inputs, targets = prepare_batch(batch, args)
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

                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss_sum += float(loss.item())
            num_train_batches += 1
            avg_train_loss_local = train_loss_sum / max(num_train_batches, 1)
            train_acc_local = 100.0 * correct / max(total, 1)
            if is_main_process(args):
                desc = ('[Train][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                        (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], avg_train_loss_local, train_acc_local, correct, total))
                prog_bar.set_description(desc, refresh=True)
                prog_bar.update(0)
        if is_main_process(args):
            prog_bar.close()

        global_train_loss_sum, global_correct, global_total, global_num_train_batches = reduce_metrics(
            train_loss_sum,
            correct,
            total,
            num_train_batches,
            args.device,
        )
        avg_train_loss = global_train_loss_sum / max(global_num_train_batches, 1.0)
        train_acc = 100.0 * global_correct / max(global_total, 1.0)
        best_train_loss = min(best_train_loss, avg_train_loss)
        best_train_acc = max(best_train_acc, train_acc)
        if is_main_process(args):
            write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer, head=False, train=True, test=False,
                      args=args, epoch=epoch, train_loss=avg_train_loss, correct=int(global_correct), total=int(global_total))

        lr_scheduler.step()

        net.eval()
        test_loss_sum = 0.0
        correct = 0
        total = 0
        num_test_batches = 0
        desc = ('[Test][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))
        if is_main_process(args):
            prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
        else:
            prog_bar = enumerate(testloader)

        batch_idx = -1
        with torch.no_grad():
            for batch_idx, batch in prog_bar:
                if args.dataset in GENERATIVE_TEXT_DATASETS:
                    batch_inputs, metadata = prepare_batch(batch, args)
                    model_inputs = {
                        'input_ids': batch_inputs['input_ids'],
                        'attention_mask': batch_inputs['attention_mask'],
                        'labels': batch_inputs['labels'],
                    }
                    outputs = forward_outputs(net, model_inputs)
                    loss = outputs.loss
                    batch_correct, batch_total = evaluate_gsm8k_generation(
                        net,
                        batch_inputs,
                        metadata,
                        generation_tokenizer,
                        args,
                    )
                    correct += batch_correct
                    total += batch_total
                else:
                    inputs, targets = prepare_batch(batch, args)
                    logits = forward_logits(net, inputs)
                    loss = criterion(logits, targets)
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                test_loss_sum += float(loss.item())
                num_test_batches += 1
                avg_test_loss_local = test_loss_sum / max(num_test_batches, 1)
                test_acc_local = 100.0 * correct / max(total, 1)
                if is_main_process(args):
                    desc = ('[Test][%s][%s][LR=%.6f] Loss: %.4f | Acc: %.3f%% (%d/%d)' %
                            (optim_name, epoch + 1, lr_scheduler.get_last_lr()[0], avg_test_loss_local, test_acc_local, correct, total))
                    prog_bar.set_description(desc, refresh=True)
        if is_main_process(args):
            prog_bar.close()

        global_test_loss_sum, global_correct, global_total, global_num_test_batches = reduce_metrics(
            test_loss_sum,
            correct,
            total,
            num_test_batches,
            args.device,
        )
        avg_test_loss = global_test_loss_sum / max(global_num_test_batches, 1.0)
        acc = 100.0 * global_correct / max(global_total, 1.0)
        best_test_loss = min(best_test_loss, avg_test_loss)
        best_test_acc = max(best_test_acc, acc)
        if is_main_process(args):
            write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer, head=False, train=False, test=True,
                      args=args, epoch=epoch, test_loss=avg_test_loss, acc=acc, train_loss=avg_train_loss)

        epoch_wallclock = time.time() - epoch_start
        cumulative_wallclock += epoch_wallclock
        if is_main_process(args):
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
            state_dict = get_model_state_dict(net, args)
            if is_main_process(args):
                best_state = {
                    'net': state_dict,
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

    close_loggers(csv_handles, wallclock_handles)

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

    if is_main_process(args):
        sweep_csv, sweep_writer = prepare_sweep_csv(
            args.log_path,
            args.dataset,
            args.network,
            args.depth,
            sweep_optimizer,
            args.experiment_type,
        )
    else:
        sweep_csv, sweep_writer = None, None

    for combo_index, combo in enumerate(combos, start=1):
        run_args = apply_sweep_combo(copy.deepcopy(args), sweep_optimizer, combo)
        run_args.resume = False
        extra_tag = format_sweep_tag(combo_index, combo)
        if is_main_process(args):
            print(f'Running sweep combo {combo_index}/{len(combos)}: {json.dumps(combo, default=str)}')
        metrics = train_one_run(run_args, extra_tag=extra_tag)
        if is_main_process(args):
            write_sweep_result(sweep_csv, sweep_writer, combo_index, combo, metrics)

    if sweep_csv is not None:
        sweep_csv.close()


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    return normalize_args(args)


def run_with_args(args):
    if args.run_sweep:
        run_sweep(args)
        return None
    metrics = train_one_run(args)
    return metrics['best_test_accuracy']


def _distributed_worker(local_rank, args):
    try:
        setup_distributed(args, rank=local_rank, world_size=args.gpu_numbers)
        run_with_args(args)
    finally:
        cleanup_distributed()


def main():
    args = parse_args()

    if not is_distributed_training(args):
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return run_with_args(args)

    if args.gpu_numbers <= 0:
        raise ValueError('--gpu_numbers must be a positive integer when --distributed_training=True.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required when --distributed_training=True.')
    if args.gpu_numbers > torch.cuda.device_count():
        raise ValueError(
            f'gpu_numbers={args.gpu_numbers} exceeds the number of visible CUDA devices ({torch.cuda.device_count()}).'
        )

    if args.gpu_numbers == 1:
        try:
            setup_distributed(args, rank=0, world_size=1)
            return run_with_args(args)
        finally:
            cleanup_distributed()

    mp.spawn(_distributed_worker, nprocs=args.gpu_numbers, args=(args,), join=True)
    return None


if __name__ == '__main__':
    main()





# CUDA_VISIBLE_DEVICES=0,1,2,3
# /root/projects/Efficient-Natural-Gradient-Descent-with-Theoretical-Guarantees
# /root/autodl-tmp
# python main.py --dataset cifar10 --network resnet --depth 20 --text_model_name distilbert/distilbert-base-uncased --run_sweep --sweep_optimizer dngd --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6],"damping":[3e-1,3e-2,3e-3,3e-4,3e-5,6e-1,6e-2,6e-3,6e-4,6e-5],"momentum":[0.3,0.6,0.9]}' --batch_size 32 --epoch 20 --device cuda:0 --data_path /root/autodl-tmp --num_workers 2 --distributed_training True --gpu_numbers 2
# python main.py --dataset gsm8k --data_path /root/autodl-tmp --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --optimizer dngd --learning_rate 3e-5 --weight_decay 5e-4 --momentum 0.6 --damping 6e-4 --batch_size 32 --eval_batch_size 32 --max_seq_length 512 --generation_max_new_tokens 256 --epoch 1 --num_workers 2 --distributed_training True --gpu_numbers 2# python main.py --dataset gsm8k --data_path /root/autodl-tmp --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --optimizer adamw --learning_rate 3e-5 --eps 1e-3 --weight_decay 5e-4 --batch_size 32 --eval_batch_size 32 --max_seq_length 512 --generation_max_new_tokens 256 --epoch 1 --num_workers 2 --distributed_training True --gpu_numbers 2
# python main.py --dataset gsm8k --data_path /root/autodl-tmp --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --optimizer adam_ --learning_rate 1e-3 --weight_decay 5e-4 --batch_size 32 --eval_batch_size 32 --max_seq_length 512 --generation_max_new_tokens 256 --epoch 1 --num_workers 2 --distributed_training True --gpu_numbers 2
# python main.py --dataset gsm8k --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --optimizer adam_ --learning_rate 1e-5 --batch_size 1 --eval_batch_size 1 --epoch 1 --max_seq_length 512 --generation_max_new_tokens 256 --device cuda