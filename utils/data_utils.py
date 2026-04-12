import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, SVHN


def _load_tinyimagenet_val_annotations(val_anno_path):
    annotations_map = {}
    with open(val_anno_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name, wnid = parts[0], parts[1]
                annotations_map[img_name] = wnid
    return annotations_map


class TinyImageNetValDataset(Dataset):
    def __init__(self, annotations_map, img_dir, class_to_idx, transform=None):
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = list(annotations_map.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, wnid = self.samples[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[wnid]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class NoiseDataset(Dataset):
    def __init__(self, dataset, noise_rate=0.2, noise_mode='sym', root_dir='./data',
                 transform=None, mode='train'):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.noise_mode = noise_mode
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        if dataset == 'cifar10':
            train_dataset = CIFAR10(root=root_dir, train=True, download=True)
            test_dataset = CIFAR10(root=root_dir, train=False, download=True)
        elif dataset == 'cifar100':
            train_dataset = CIFAR100(root=root_dir, train=True, download=True)
            test_dataset = CIFAR100(root=root_dir, train=False, download=True)
        else:
            raise ValueError(f'NoiseDataset only supports cifar10/cifar100, got {dataset}')

        self.train_data = train_dataset.data
        self.train_label = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_label = np.array(test_dataset.targets)

        if noise_rate > 0:
            self.noise_label = self._inject_noise(self.train_label, noise_rate, noise_mode)
        else:
            self.noise_label = self.train_label.copy()

    def _inject_noise(self, labels, noise_rate, noise_mode):
        labels = labels.copy()
        n = len(labels)
        num_noisy = int(noise_rate * n)
        noisy_idx = np.random.choice(n, num_noisy, replace=False)
        num_classes = len(np.unique(labels))

        if noise_mode == 'sym':
            for i in noisy_idx:
                old = labels[i]
                new = np.random.randint(num_classes)
                while new == old:
                    new = np.random.randint(num_classes)
                labels[i] = new
        elif noise_mode == 'asym':
            for i in noisy_idx:
                labels[i] = (labels[i] + 1) % num_classes
        else:
            raise ValueError(f'Unsupported noise mode: {noise_mode}')

        return labels

    def __getitem__(self, index):
        if self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
        else:
            img, target = self.train_data[index], self.noise_label[index]

        img = Image.fromarray(img)
        img = self.transform(img) if self.transform else img
        return img, target

    def __len__(self):
        return len(self.test_data) if self.mode == 'test' else len(self.train_data)


class SST2Dataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_seq_length=128):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, index):
        example = self.hf_split[index]
        encoded = self.tokenizer(
            example['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        inputs = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
        if 'token_type_ids' in encoded:
            inputs['token_type_ids'] = encoded['token_type_ids'].squeeze(0)

        label = int(example['label'])
        return inputs, label


class GSM8KDataset(Dataset):
    PROMPT_TEMPLATE = 'Question:\n{question}\n\nAnswer:\n'

    def __init__(self, hf_split, tokenizer, max_seq_length=512, mode='train'):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    @staticmethod
    def extract_final_answer(answer_text):
        answer_text = str(answer_text).strip()
        if '####' in answer_text:
            return answer_text.split('####')[-1].strip().replace(',', '')
        return answer_text.replace(',', '').strip()

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, index):
        example = self.hf_split[index]
        question = str(example['question']).strip()
        answer = str(example['answer']).strip()

        prompt = self.PROMPT_TEMPLATE.format(question=question)
        full_text = prompt + answer

        prompt_only = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        prompt_length = min(int(prompt_only['attention_mask'].sum().item()), labels.size(0))
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        if self.mode != 'train':
            original_padding_side = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = 'left'
                generation_prompt = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_length,
                    return_tensors='pt',
                )
            finally:
                self.tokenizer.padding_side = original_padding_side
            item['generation_input_ids'] = generation_prompt['input_ids'].squeeze(0)
            item['generation_attention_mask'] = generation_prompt['attention_mask'].squeeze(0)
            item['answer_text'] = answer
            item['final_answer'] = self.extract_final_answer(answer)
        # if self.mode != 'train':
        #     generation_prompt = self.tokenizer(
        #         prompt,
        #         truncation=True,
        #         max_length=self.max_seq_length,
        #         return_tensors='pt',
        #     )
        #     item['generation_input_ids'] = generation_prompt['input_ids'].squeeze(0)
        #     item['generation_attention_mask'] = generation_prompt['attention_mask'].squeeze(0)
        #     item['answer_text'] = answer
        #     item['final_answer'] = self.extract_final_answer(answer)

        return item


class NoiseDataLoader:
    def __init__(
        self,
        dataset,
        noise_rate=0.2,
        noise_mode='sym',
        batch_size=128,
        num_workers=4,
        root_dir='./data',
        text_model_name='distilbert-base-uncased',
        max_seq_length=128,
        eval_batch_size=None,
        gsm8k_config='main',
    ):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.text_model_name = text_model_name
        self.max_seq_length = max_seq_length
        self.eval_batch_size = eval_batch_size
        self.gsm8k_config = gsm8k_config

        if noise_rate <= 0:
            self.noise_mode = None

    def get_transforms(self):
        if self.dataset in ['fashionmnist', 'mnist']:
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])
        elif self.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        elif self.dataset == 'svhn':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
        elif self.dataset == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif self.dataset == 'miniimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif self.dataset == 'tinyimagenet':
            mean = (0.4802, 0.4481, 0.3975)
            std = (0.2302, 0.2265, 0.2262)
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif self.dataset in {'sst2', 'gsm8k'}:
            transform_train = None
            transform_test = None
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset}')
        return transform_train, transform_test

    def get_loader(self, distributed_training=False, rank=0, world_size=1):
        transform_train, transform_test = self.get_transforms()
        print(f'Loading {self.dataset} dataset.')

        if self.dataset in ['fashionmnist', 'mnist']:
            dataset_cls = FashionMNIST if self.dataset == 'fashionmnist' else MNIST
            train_dataset = dataset_cls(root=self.root_dir, train=True, download=True, transform=transform_train)
            test_dataset = dataset_cls(root=self.root_dir, train=False, download=True, transform=transform_test)
        elif self.dataset in ['cifar10', 'cifar100']:
            if self.noise_mode is None:
                print('Going without noise.')
                dataset_cls = CIFAR10 if self.dataset == 'cifar10' else CIFAR100
                train_dataset = dataset_cls(root=self.root_dir, train=True, download=True, transform=transform_train)
                test_dataset = dataset_cls(root=self.root_dir, train=False, download=True, transform=transform_test)
            else:
                train_dataset = NoiseDataset(
                    dataset=self.dataset,
                    noise_rate=self.noise_rate,
                    noise_mode=self.noise_mode,
                    root_dir=self.root_dir,
                    transform=transform_train,
                    mode='train',
                )
                test_dataset = NoiseDataset(
                    dataset=self.dataset,
                    noise_rate=self.noise_rate,
                    noise_mode=self.noise_mode,
                    root_dir=self.root_dir,
                    transform=transform_test,
                    mode='test',
                )
        elif self.dataset == 'svhn':
            train_dataset = SVHN(root=self.root_dir, split='train', download=True, transform=transform_train)
            extra_dataset = SVHN(root=self.root_dir, split='extra', download=True, transform=transform_train)
            train_dataset = ConcatDataset([train_dataset, extra_dataset])
            test_dataset = SVHN(root=self.root_dir, split='test', download=True, transform=transform_test)
        elif self.dataset == 'imagenet':
            traindir = os.path.join(self.root_dir, 'imagenet', 'train')
            valdir = os.path.join(self.root_dir, 'imagenet', 'val')
            train_dataset = datasets.ImageFolder(traindir, transform_train)
            test_dataset = datasets.ImageFolder(valdir, transform_test)
        elif self.dataset == 'miniimagenet':
            train_dir = os.path.join(self.root_dir, 'miniimagenet', 'train')
            test_dir = os.path.join(self.root_dir, 'miniimagenet', 'test')
            train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
            test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
        elif self.dataset == 'tinyimagenet':
            base_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
            train_dir = os.path.join(base_dir, 'train')
            val_img_dir = os.path.join(base_dir, 'val', 'images')
            val_anno = os.path.join(base_dir, 'val', 'val_annotations.txt')
            train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
            class_to_idx = train_dataset.class_to_idx
            annotations_map = _load_tinyimagenet_val_annotations(val_anno)
            test_dataset = TinyImageNetValDataset(
                annotations_map=annotations_map,
                img_dir=val_img_dir,
                class_to_idx=class_to_idx,
                transform=transform_test,
            )
        elif self.dataset == 'sst2':
            try:
                from datasets import load_dataset
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    'SST-2 support requires `datasets` and `transformers`. Please install them first: pip install datasets transformers'
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            glue_dataset = load_dataset('glue', 'sst2')
            train_dataset = SST2Dataset(glue_dataset['train'], tokenizer=tokenizer, max_seq_length=self.max_seq_length)
            test_dataset = SST2Dataset(glue_dataset['validation'], tokenizer=tokenizer, max_seq_length=self.max_seq_length)
        elif self.dataset == 'gsm8k':
            try:
                from datasets import load_dataset
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    'GSM8K support requires `datasets` and `transformers`. Please install them first: pip install datasets transformers'
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.padding_side = 'right'

            gsm8k_dataset = load_dataset('openai/gsm8k', self.gsm8k_config)
            train_dataset = GSM8KDataset(
                gsm8k_dataset['train'],
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                mode='train',
            )
            test_dataset = GSM8KDataset(
                gsm8k_dataset['test'],
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                mode='eval',
            )
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset}')

        eval_batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
        if self.dataset == 'gsm8k' and self.eval_batch_size is None:
            eval_batch_size = 1

        train_sampler = None
        test_sampler = None
        train_shuffle = True
        test_shuffle = False
        if distributed_training:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            train_shuffle = False
            test_shuffle = False

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            shuffle=test_shuffle,
            sampler=test_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, test_loader
