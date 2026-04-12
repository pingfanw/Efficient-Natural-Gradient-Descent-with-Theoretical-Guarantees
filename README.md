# Efficient-Natural-Gradient-Descent-with-Theoretical-Guarantees
Numerial Experiment of the paper "Diagonal Natural Gradient Descent: Training Efficiency Improved with Generalization Preserved".
Numerial Experiment of the paper "Diagonal Natural Gradient Descent: Training Efficiency Improved with Generalization Preserved".

Requirements:

numpy=1.26.0
Pillow>=9.5.0
tqdm>=4.66.0
timm>=0.9.0
transformers>=4.51.0
datasets>=2.16.0
torch=2.5.1+cu118(or higher)
torchvision=0.20.1+cu118(or higher)
python=3.10.4

## 1. Overview

1. **Task Support**:

	- Image Classification: `mnist`, `fashionmnist`, `cifar10`, `cifar100`, `svhn`, `imagenet`, `miniimagenet`, `tinyimagenet`
	- Text Classification: `sst2`
	- Mathematical Reasoning Generation Task: `gsm8k`

2. **Model Support**:

	- Classic CNN: `alexnet`, `vgg16_bn`, `vgg19_bn`, `resnet`, `wrn`, `preresnet`, `densenet`, `pyramidnet`
	- MLP: `mlp`
	- Vision Transformer: `vit_small`, `vit_base`, `vit_large`, `vit_huge`
	- Hugging Face Pretrained Vision Models: `pretrain_vit`, `pretrain_swin`, `pretrain_deit`, `pretrain_beit`, `pretrain_pvt`
	- Hugging Face Text Models: `distilbert_sst2`, `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_8b`

3. **Optimizer Support**:

	- Standard Optimizers: `sgd`
	- Custom Optimizers: `sgd_`, `adam_`, `adagrad_`, `adamw`, `muon`
	- Natural Gradient / Approximate Natural Gradient Classes: `dngd`, `ngd`, `kfac`, `ekfac`
	- Sharpness-Aware Optimization: `sam`

4. **Experimental Features**:

	- Single Training Run
	- Hyperparameter Grid Search
	- DDP Distributed Training (sam not supported currently)
	- Checkpoint Resumption
	- CSV Logging
	- Best Checkpoint Saving

5. **Data Flow and Training Pipeline**

	Read parameters.

	Create corresponding `DataLoader` based on `dataset`.

	Instantiate model based on `dataset + network`.

	Build optimizer based on `optimizer`.

	Build learning rate scheduler based on `milestone` or default rules.

	For each epoch:

		Training phase: forward, backward, parameter update.
		
		Testing phase:
		
			Calculate accuracy for classification tasks;
		
			`gsm8k` generates answers and calculates exact match.
		
			Record train/test/wallclock CSV.
		
			Save best checkpoint if test metric improves.

---

## 2. Parameter Overview

### 2.1 Data and Task-Related Parameters

| Parameter                     |                              Default | Purpose                                          | Practical Explanation                                        |
| ----------------------------- | -----------------------------------: | ------------------------------------------------ | ------------------------------------------------------------ |
| `--dataset`                   |                            `cifar10` | Specifies task/dataset                           | Supports `cifar10,cifar100,mnist,fashionmnist,svhn,imagenet,miniimagenet,tinyimagenet,sst2,gsm8k`. Will be normalized to lowercase in `get_networks.py`. |
| `--noise_rate`                |                                `0.0` | Label noise ratio                                | Only effective for `NoiseDataset` of `cifar10/cifar100`. Other datasets essentially do not use this value, but log naming may still include it. |
| `--noise_mode`                |                                `sym` | Noise type                                       | `sym` indicates symmetric noise, `asym` indicates asymmetric noise injected as `(y+1) mod C`. Only effective in `cifar10/cifar100` noise experiments. |
| `--num_workers`               |                                  `0` | Number of `DataLoader` workers                   | Available for all datasets.                                  |
| `--data_path`                 |                        `E:\datasets` | Local image data root directory                  | Effective for `MNIST/CIFAR/SVHN/ImageNet/MiniImageNet/TinyImageNet`. Does not actually control HF dataset cache location for `sst2/gsm8k`. |
| `--outputs_dim`               |                                 `10` | Number of output classes                         | Automatically overridden by `get_networks.py` based on `dataset`. |
| `--input_dim`                 |                                 `32` | Input dimension size                             | Automatically overridden by `get_networks.py` based on `dataset`. |
| `--in_channels`               |                                  `3` | Number of input channels                         | Automatically overridden by `get_networks.py` based on `dataset`. |
| `--max_seq_length`            |                                `128` | Maximum text sequence length                     | Effective for `sst2` and `gsm8k`. `gsm8k` often needs to be increased to 512. |
| `--text_model_name`           | `distilbert/distilbert-base-uncased` | Hugging Face model name                          | Effective for `sst2` / `gsm8k`, also passed to pretrained text model wrappers. |
| `--eval_batch_size`           |                               `None` | Test batch size                                  | If not set, defaults to `--batch_size`; **but for `gsm8k`, if not explicitly set, defaults to test batch=1**. |
| `--gsm8k_config`              |                               `main` | Hugging Face GSM8K config name                   | Optional `main` or `socratic`. Only effective for `gsm8k`.   |
| `--generation_max_new_tokens` |                                `256` | Max new tokens generated during GSM8K evaluation | Only effective during `gsm8k` testing/generation evaluation. |

### 2.2 Model Architecture-Related Parameters

| Parameter           | Default Value | Purpose                        | Practical Explanation                                        |
| ------------------- | ------------: | ------------------------------ | ------------------------------------------------------------ |
| `--network`         |    `vgg16_bn` | Model name                     | Must match models supported by the current `dataset`.        |
| `--depth`           |          `16` | Model depth                    | Primarily used for structures like `resnet/preresnet/densenet/wrn/pyramidnet`; VGG/text models typically do not depend on it. |
| `--growthRate`      |          `12` | DenseNet growth rate           | Only used by DenseNet class models.                          |
| `--compressionRate` |           `2` | DenseNet compression rate      | Only used by DenseNet class models.                          |
| `--widen_factor`    |           `1` | WRN widen factor               | Only used by WRN class models.                               |
| `--dropRate`        |         `0.1` | Dropout rate                   | Commonly used in WRN / DenseNet implementations.             |
| `--mlp_ratio`       |         `4.0` | ViT MLP hidden expansion ratio | Only effective for `vit_small/base/large/huge`.              |
| `--patch_size`      |          `16` | ViT patch size                 | Only effective for ViT models.                               |

### 2.3 Optimization and Training-Related Parameters

| Parameter              |  Default Value | Purpose                                                  | Practical Explanation                                        |
| ---------------------- | -------------: | -------------------------------------------------------- | ------------------------------------------------------------ |
| `--optimizer`          |        `adam_` | Optimizer name                                           | See the optimizer list above for possible values.            |
| `--learning_rate`      |          `0.1` | Learning rate                                            | Used by all optimizers.                                      |
| `--batch_size`         |          `128` | Training batch size                                      | Under DDP, represents **global batch size**; code converts to local batch based on GPU count. |
| `--epoch`              |          `300` | Number of training epochs                                | Common to all tasks.                                         |
| `--eps`                |        `1e-10` | Numerical stability term                                 | Primarily used by `adam_`, `adagrad_`, `adamw`, `muon`.      |
| `--retraction_eps`     |        `1e-10` | Muon retraction stability term                           | Only used by `muon`.                                         |
| `--amsgrad`            |        `False` | Whether to enable AMSGrad                                | Only used by `adam_`; type parsing supports `True/False/yes/no/1/0`. |
| `--betas`              | `(0.9, 0.999)` | Adam/Muon class first/second-order momentum coefficients | Used by `adam_`, `adamw`, `muon`, and some base optimizers of `sam`. Can be written as `"(0.9, 0.999)"`. |
| `--milestone`          |         `None` | Milestones for MultiStepLR                               | Format like `"50,100,150"`. If not set: ViT defaults to predefined MultiStep, other models default to `CosineAnnealingLR`. |
| `--momentum`           |          `0.9` | Momentum                                                 | Used by `sgd/sgd_/dngd/ngd/kfac/ekfac`, and some base optimizers of `sam`. |
| `--stat_decay`         |         `0.95` | Covariance statistics decay factor                       | Only used by `kfac/ekfac`.                                   |
| `--damping`            |         `1e-3` | Damping term                                             | Used by `dngd/ngd/kfac/ekfac`.                               |
| `--kl_clip`            |         `1e-2` | KFAC/EKFAC KL clip                                       | Only used by `kfac/ekfac`.                                   |
| `--weight_decay`       |         `5e-4` | Weight decay                                             | Used by most optimizers.                                     |
| `--rho`                |         `0.05` | SAM neighborhood radius                                  | Only used by `sam`.                                          |
| `--sam_base_optimizer` |          `sgd` | SAM base optimizer                                       | Only used by `sam`. Allows `sgd/sgd_/adam_/adagrad_/adamw`.  |
| `--TCov`               |          `100` | Covariance statistics update period                      | Only used by `kfac/ekfac`.                                   |
| `--TScal`              |          `100` | EKFAC scaling statistics update period                   | Only used by `ekfac`.                                        |
| `--TInv`               |          `100` | Inverse matrix update period                             | Only used by `kfac/ekfac`.                                   |

### 2.4 Sweep (Grid Search) Parameters

| Parameter           | Default Value | Purpose                                       | Practical Explanation                                        |
| ------------------- | ------------: | --------------------------------------------- | ------------------------------------------------------------ |
| `--run_sweep`       |       `False` | Whether to perform hyperparameter grid search | When enabled, skips single training run and enumerates Cartesian product of `sweep_params`. |
| `--sweep_optimizer` |        `None` | Specifies optimizer for sweep                 | If `None`, falls back to `--optimizer`.                      |
| `--sweep_params`    |          `''` | Sweep parameter dictionary                    | Supports JSON or Python dict string, e.g., `'{"learning_rate":[1e-3,1e-4],"weight_decay":[0,1e-4]}'`. |

Note:

- Each optimizer only allows sweeping certain parameters, controlled by `OPTIMIZER_SWEEP_PARAMS` in `utils/get_optimizers.py`.
- Sweep results are additionally written to `logs/.../sweep/<optimizer>_summary.csv`.

### 2.5 Device, Distributed, Logging, and Resumption Parameters

| Parameter                | Default Value | Purpose                           | Practical Explanation                                        |
| ------------------------ | ------------: | --------------------------------- | ------------------------------------------------------------ |
| `--device`               |        `cuda` | Single-card training device       | Used directly for single card; automatically rewritten as `cuda:{rank}` under DDP. |
| `--distributed_training` |       `False` | Whether to enable DDP             | Uses `torch.multiprocessing.spawn + NCCL` when enabled.      |
| `--gpu_numbers`          |           `1` | Number of GPUs for DDP            | Only effective when `distributed_training=True`.             |
| `--resume` / `-r`        |       `False` | Whether to resume from checkpoint | When enabled, reads weights and epoch from `--load_path`.    |
| `--load_path`            |          `''` | Checkpoint path                   | Used by `--resume`.                                          |
| `--log_path`             |      `./logs` | Root log directory                | train/test/wallclock/sweep CSV files are written here.       |
| `--experiment_type`      |       `error` | Experiment category label         | Affects directory/file naming for logs and checkpoints, options: `error/noise/damping`. |

---

## 3. Logging, Output, and Checkpoints

### 3.1 Log Directory Structure

Log directory root is `--log_path`, structure resembles:

```text
logs/
└── <experiment_type>/
    └── <dataset>/
        └── <model or model+depth>/
            ├── train/
            ├── test/
            ├── wallclock/
            └── sweep/
```

Where:

- `train/*.csv`: Training loss and training accuracy (or token accuracy) per epoch
- `test/*.csv`: Test loss and test accuracy (or GSM8K exact match) per epoch
- `wallclock/*.csv`: Single-round time and cumulative time per epoch
- `sweep/*_summary.csv`: Summary of best metrics for each hyperparameter combination

### 3.2 Checkpoint Saving Rules

Best checkpoint is only saved under the following conditions:

1. `acc > best_acc`
2. `experiment_type == 'error'`

Save directory root is specified by `--model_path`, path structure resembles:

```text
pretrain/error/<dataset>/<network or network+depth>/
```

---

## 4. Dataset Directory Requirements

### 4.1 Automatically Downloaded Datasets

The following datasets are automatically downloaded by torchvision or Hugging Face:

- `mnist`
- `fashionmnist`
- `cifar10`
- `cifar100`
- `svhn`
- `sst2`
- `gsm8k`

### 4.2 Datasets Requiring Local Directory Organization

#### ImageNet

```text
<data_path>/imagenet/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

#### MiniImageNet

```text
<data_path>/miniimagenet/
├── train/
│   ├── class1/
│   └── ...
└── test/
    ├── class1/
    └── ...
```

#### TinyImageNet

```text
<data_path>/tiny-imagenet-200/
├── train/
├── val/
│   ├── images/
│   └── val_annotations.txt
```

---

## 5. Execution Examples

### 5.1 Single Image Classification Training (CIFAR-10 + ResNet)

```bash
python main.py \
  --dataset cifar10 \
  --network resnet \
  --depth 20 \
  --optimizer sgd \
  --learning_rate 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --batch_size 128 \
  --epoch 200 \
  --data_path ./data \
  --device cuda:0
```

### 5.2 CIFAR-10 Label Noise Experiment

```bash
python main.py \
  --dataset cifar10 \
  --network resnet \
  --depth 20 \
  --optimizer dngd \
  --learning_rate 0.01 \
  --momentum 0.9 \
  --damping 1e-3 \
  --noise_rate 0.2 \
  --noise_mode sym \
  --experiment_type noise \
  --batch_size 128 \
  --epoch 200 \
  --data_path ./data \
  --device cuda:0
```

### 5.3 SST-2 Text Classification

```bash
python main.py \
  --dataset sst2 \
  --network distilbert_sst2 \
  --text_model_name distilbert/distilbert-base-uncased \
  --optimizer adamw \
  --learning_rate 2e-5 \
  --weight_decay 1e-2 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 128 \
  --epoch 3 \
  --device cuda:0
```

### 5.4 GSM8K + Qwen3 Single Training

```bash
python main.py \
  --dataset gsm8k \
  --network qwen3_0_6b \
  --text_model_name Qwen/Qwen3-0.6B-Base \
  --optimizer adamw \
  --learning_rate 3e-5 \
  --betas "(0.9, 0.999)" \
  --eps 1e-3 \
  --weight_decay 5e-4 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 512 \
  --generation_max_new_tokens 256 \
  --epoch 1 \
  --num_workers 2 \
  --distributed_training True \
  --gpu_numbers 2
```

### 5.5 GSM8K Hyperparameter Sweep (AdamW)

```bash
python main.py \
  --dataset gsm8k \
  --network qwen3_0_6b \
  --text_model_name Qwen/Qwen3-0.6B-Base \
  --run_sweep \
  --sweep_optimizer adamw \
  --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6],"betas":[[0.5,0.55],[0.7,0.77],[0.9,0.99]],"eps":[1e-1,1e-2,1e-3],"weight_decay":[5e-4,5e-6,5e-2]}' \
  --epoch 3 \
  --batch_size 64 \
  --eval_batch_size 64 \
  --max_seq_length 512 \
  --generation_max_new_tokens 256 \
  --distributed_training True \
  --num_workers 2 \
  --gpu_numbers 2
```

Note:

- `sweep_params` can be written in JSON or Python dict style.
- `betas` will be automatically normalized to tuple during sweep.

### 5.6 Distributed Training (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --dataset cifar10 \
  --network resnet \
  --depth 20 \
  --optimizer sgd \
  --learning_rate 0.1 \
  --batch_size 256 \
  --epoch 200 \
  --distributed_training True \
  --gpu_numbers 2 \
  --data_path ./data
```

Here `batch_size=256` represents **global batch size**. The code will automatically calculate per-GPU batch size.

### 5.7 Resume Training from Checkpoint

```bash
python main.py \
  --dataset cifar10 \
  --network resnet \
  --depth 20 \
  --optimizer sgd \
  --resume \
  --load_path ./pretrain/error/cifar10/resnet20/sgd_cifar10_resnet20_best.t7 \
  --epoch 200 \
  --device cuda:0
```