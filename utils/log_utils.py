import csv
import json
import os

TEXT_DATASETS = {'sst2'}
VIT_NETWORKS = {'vit_small', 'vit_base', 'vit_large', 'vit_huge'}

def _model_dir_name(dataset, model, depth):
    if model in VIT_NETWORKS or dataset in TEXT_DATASETS:
        return model
    return f'{model}{depth}'


def _build_log_dir(log_path, experiment_type, dataset, model, depth, split_name):
    base_dir = os.path.join(log_path, experiment_type, dataset, _model_dir_name(dataset, model, depth), split_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _build_file_stem(optimizer, noise_rate, damping, experiment_type, extra_tag=None):
    if experiment_type == 'error' and noise_rate == 0.0:
        stem = optimizer.lower()
    elif experiment_type == 'damping' and noise_rate == 0.0:
        stem = f'{optimizer.lower()}_damping{damping}'
    else:
        stem = f'{optimizer.lower()}_noise{noise_rate}'
    if extra_tag:
        stem = f'{stem}_{extra_tag}'
    return stem


def prepare_csv(log_path, dataset, model, depth, optimizer, noise_rate, damping, experiment_type, extra_tag=None):
    train_log_path = _build_log_dir(log_path, experiment_type, dataset, model, depth, 'train')
    test_log_path = _build_log_dir(log_path, experiment_type, dataset, model, depth, 'test')
    file_stem = _build_file_stem(optimizer, noise_rate, damping, experiment_type, extra_tag=extra_tag)
    csv_train = open(os.path.join(train_log_path, file_stem + '.csv'), 'a+', newline='')
    csv_test = open(os.path.join(test_log_path, file_stem + '.csv'), 'a+', newline='')
    return csv_train, csv.writer(csv_train), csv_test, csv.writer(csv_test)


def prepare_wallclock_csv(log_path, dataset, model, depth, optimizer, noise_rate, damping, experiment_type, extra_tag=None):
    wallclock_log_path = _build_log_dir(log_path, experiment_type, dataset, model, depth, 'wallclock')
    file_stem = _build_file_stem(optimizer, noise_rate, damping, experiment_type, extra_tag=extra_tag)
    wallclock_csv = open(os.path.join(wallclock_log_path, file_stem + '.csv'), 'a+', newline='')
    return wallclock_csv, csv.writer(wallclock_csv)


def prepare_sweep_csv(log_path, dataset, model, depth, optimizer, experiment_type):
    sweep_log_path = _build_log_dir(log_path, experiment_type, dataset, model, depth, 'sweep')
    sweep_csv = open(os.path.join(sweep_log_path, optimizer.lower() + '_summary.csv'), 'a+', newline='')
    return sweep_csv, csv.writer(sweep_csv)


def write_csv(csv_train, csv_train_writer, csv_test, csv_test_writer,
              head=False, train=False, test=False, args=None, **kwargs):
    if head:
        if args is None:
            raise ValueError('Parser is None! Please check!')
        csv_train_writer.writerow([
            'network:', args.network, 'depth', args.depth, 'Loss:CrossEntropy', 'Dataset:', args.dataset, 'Optimizer:', kwargs['optim_name'],
            'LearningRate:', args.learning_rate, 'BatchSize:', args.batch_size, 'EpochRange:', args.epoch,
        ])
        csv_train_writer.writerow(['Epoch', 'Train_Loss', 'Train_Accuracy'])
        csv_train.flush()

        csv_test_writer.writerow([
            'network:', args.network, 'depth', args.depth, 'Loss:CrossEntropy', 'Dataset:', args.dataset, 'Optimizer:', kwargs['optim_name'],
            'LearningRate:', args.learning_rate, 'BatchSize:', args.batch_size, 'EpochRange:', args.epoch,
        ])
        csv_test_writer.writerow(['Epoch', 'Test_Loss', 'Test_Accuracy', 'Generalization_Gap'])
        csv_test.flush()
    elif train:
        csv_train_writer.writerow([
            kwargs['epoch'] + 1,
            kwargs['train_loss'],
            100.0 * kwargs['correct'] / max(kwargs['total'], 1),
        ])
        csv_train.flush()
    elif test:
        csv_test_writer.writerow([
            kwargs['epoch'] + 1,
            kwargs['test_loss'],
            kwargs['acc'],
            abs(kwargs['test_loss'] - kwargs['train_loss']),
        ])
        csv_test.flush()
    else:
        raise ValueError('head, train, test are all False! Nothing to write!')


def write_wallclock_csv(wallclock_csv, wallclock_writer, head=False, args=None, **kwargs):
    if head:
        if args is None:
            raise ValueError('Parser is None! Please check!')
        wallclock_writer.writerow([
            'network:', args.network, 'depth', args.depth, 'Dataset:', args.dataset, 'Optimizer:', kwargs['optim_name'],
            'LearningRate:', args.learning_rate, 'BatchSize:', args.batch_size, 'EpochRange:', args.epoch,
        ])
        wallclock_writer.writerow(['Epoch', 'Epoch_Wallclock_Sec', 'Cumulative_Wallclock_Sec'])
        wallclock_csv.flush()
        return

    wallclock_writer.writerow([
        kwargs['epoch'] + 1,
        round(kwargs['epoch_wallclock_sec'], 6),
        round(kwargs['cumulative_wallclock_sec'], 6),
    ])
    wallclock_csv.flush()


def write_sweep_result(sweep_csv, sweep_writer, combo_index, combo, metrics):
    if sweep_csv.tell() == 0:
        sweep_writer.writerow([
            'Combo_Index',
            'Optimizer',
            'Hyperparameters',
            'Best_Train_Loss',
            'Best_Test_Loss',
            'Best_Train_Accuracy',
            'Best_Test_Accuracy',
            'Wallclock_Sec',
        ])
    sweep_writer.writerow([
        combo_index,
        metrics['optimizer'],
        json.dumps(combo, sort_keys=True),
        metrics['best_train_loss'],
        metrics['best_test_loss'],
        metrics['best_train_accuracy'],
        metrics['best_test_accuracy'],
        metrics['wallclock_sec'],
    ])
    sweep_csv.flush()