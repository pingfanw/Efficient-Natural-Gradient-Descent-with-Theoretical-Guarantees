import os
import random
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10,CIFAR100,MNIST,FashionMNIST,SVHN
from torch.utils.data import ConcatDataset
from PIL import Image

# support datasets: cifar10,cifar100,mnist,fashionmnist,imagenet,miniimagenet, tinyimagenet

def set_seed(seed=0): 
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TinyImageNetValDataset(Dataset):
    """
    Validation set for Tiny ImageNet.
    Uses val_annotations.txt to map image filename -> wnid label,
    then converts wnid into class index using trainset.class_to_idx.
    """
    def __init__(self, annotations_map, img_dir, class_to_idx, transform=None):
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_filenames = list(self.annotations_map.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        wnid = self.annotations_map[img_name]
        target = self.class_to_idx[wnid]
        if self.transform:
            image = self.transform(image)
        return image, target


def _load_tinyimagenet_val_annotations(annotations_path):
    """
    Parse tiny-imagenet-200/val/val_annotations.txt
    Returns: dict[filename] = wnid
    """
    annotations_map = {}
    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name, wnid = parts[0], parts[1]
                annotations_map[img_name] = wnid
    return annotations_map

class NoiseDataset(Dataset):
    def __init__(self,dataset,noise_rate=0.2,noise_mode='sym',root_dir='./data',
                 transform=None,mode='train'): 
        self.noise_rate = noise_rate  
        self.transform = transform
        self.mode = mode 
        self.transition = {0: 0,2: 0,4: 7,7: 7,1: 1,9: 1,3: 5,5: 3,6: 6,8: 8} 
        # Define noise file based on dataset
        noise_file_name = f"{dataset}_noise.json"
        self.noise_file = os.path.join(root_dir+'/'+'cifar-10-batches-py/'+noise_file_name if dataset == 'cifar10' else root_dir+'/'+'cifar-100-python/'+noise_file_name)
        if self.mode == 'test':
            self._load_test_data(dataset,root_dir)
        else:
            self._load_train_data(dataset,root_dir,noise_mode)

    def _load_test_data(self,dataset,root_dir):
        if dataset == 'cifar10':
            test_dic = self._unpickle(f'{root_dir}/cifar-10-batches-py/test_batch')
            self.test_data = test_dic['data'].reshape((10000,3,32,32)).transpose((0,2,3,1))
            self.test_label = test_dic['labels']
        elif dataset == 'cifar100':
            test_dic = self._unpickle(f'{root_dir}/cifar-100-python/test')
            self.test_data = test_dic['data'].reshape((10000,3,32,32)).transpose((0,2,3,1))
            self.test_label = test_dic['fine_labels']

    def _load_train_data(self,dataset,root_dir,noise_mode):
        train_data,train_label = [],[]
        if dataset == 'cifar10':
            for n in range(1,6):
                data_dic = self._unpickle(f'{root_dir}/cifar-10-batches-py/data_batch_{n}')
                train_data.append(data_dic['data'])
                train_label.extend(data_dic['labels'])
            train_data = np.concatenate(train_data)
        elif dataset == 'cifar100':
            train_dic = self._unpickle(f'{root_dir}/cifar-100-python/train')
            train_data = train_dic['data']
            train_label = train_dic['fine_labels']
        train_data = train_data.reshape((50000,3,32,32)).transpose((0,2,3,1))
        if os.path.exists(self.noise_file):
            print(f"Loading {noise_mode}metric noise of ratio {self.noise_rate} file: {self.noise_file}.")
            noise_label = json.load(open(self.noise_file,"r"))
        else:
            print(f"Injecting {noise_mode}metric noise of ratio {self.noise_rate} and saving to {self.noise_file}.")
            noise_label = self._inject_noise(train_label,dataset,noise_mode)
        if self.mode == 'train':
            self.train_data = train_data
            self.noise_label = noise_label

    def _inject_noise(self,train_label,dataset,noise_mode):
        noise_label = []
        idx = list(range(50000))
        random.shuffle(idx)
        num_noise = int(self.noise_rate * 50000)
        noise_idx = idx[:num_noise]
        for i in range(50000):
            if i in noise_idx:
                if noise_mode == 'sym':
                    # print("Symmetric noise")
                    noiselabel = random.randint(0,9) if dataset == 'cifar10' else random.randint(0,99)
                elif noise_mode == 'asym':
                    # print("Asymmetric noise")
                    noiselabel = self.transition[train_label[i]]
                noise_label.append(noiselabel)
            else:
                noise_label.append(train_label[i])
        # json.dump(noise_label,open(self.noise_file,"w"))
        return noise_label

    def _unpickle(self,file):
        import pickle as cPickle
        with open(file,'rb') as fo:
            return cPickle.load(fo,encoding='latin1')

    def __getitem__(self,index):
        if self.mode == 'test':
            img,target = self.test_data[index],self.test_label[index]
        else:
            img,target = self.train_data[index],self.noise_label[index]

        img = Image.fromarray(img)
        img = self.transform(img) if self.transform else img
        return img,target

    def __len__(self):
        return len(self.test_data) if self.mode == 'test' else len(self.train_data)

class NoiseDataLoader:
    def __init__(self,dataset,noise_rate=0.2,noise_mode='sym',batch_size=128,num_workers=4,root_dir='./data'):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        if noise_rate <= 0:
            self.noise_mode = None

    def get_transforms(self):
        if self.dataset in ['fashionmnist','mnist']:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])
        elif self.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
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
            transform_train=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                            ])
            transform_test=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                ])
        elif self.dataset == 'miniimagenet':
            transform_train = transforms.Compose([transforms.RandomResizedCrop(84),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
            transform_test = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(84),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        elif self.dataset == 'tinyimagenet':
            # Tiny ImageNet (200 classes, 64x64)
            mean = (0.4802, 0.4481, 0.3975)
            std  = (0.2302, 0.2265, 0.2262)
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
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return transform_train,transform_test

    def get_loader(self):
        transform_train,transform_test = self.get_transforms()
        print(f"Loading {self.dataset} dataset.")
        if self.dataset in ['fashionmnist','mnist']:
            dataset_cls = FashionMNIST if self.dataset == 'fashionmnist' else MNIST
            train_dataset = dataset_cls(root=self.root_dir,train=True,download=True,transform=transform_train)
            test_dataset = dataset_cls(root=self.root_dir,train=False,download=True,transform=transform_test)
        elif self.dataset in ['cifar10','cifar100']:
            if self.noise_mode == None:
                print("Going without noise.")
                if self.dataset == 'cifar10':
                    dataset_cls = CIFAR10
                else:
                    dataset_cls = CIFAR100
                train_dataset = dataset_cls(root=self.root_dir,train=True,download=True,transform=transform_train)
                test_dataset = dataset_cls(root=self.root_dir,train=False,download=True,transform=transform_test)
            else:
                train_dataset = NoiseDataset(dataset=self.dataset,
                                             noise_rate=self.noise_rate,
                                             noise_mode=self.noise_mode,
                                             root_dir=self.root_dir,
                                             transform=transform_train,
                                             mode="train")
                test_dataset = NoiseDataset(dataset=self.dataset,
                                            noise_rate=self.noise_rate,
                                            noise_mode=self.noise_mode,
                                            root_dir=self.root_dir,
                                            transform=transform_test,
                                            mode="test")
        elif self.dataset == 'svhn':
            train_dataset = SVHN(root=self.root_dir,split='train',download=True,transform=transform_train)
            extra_dataset = SVHN(root=self.root_dir,split='train',download=True,transform=transform_train)
            train_dataset = ConcatDataset([train_dataset, extra_dataset])
            test_dataset = SVHN(root=self.root_dir,split='test',download=True,transform=transform_test)
        elif self.dataset == 'imagenet':
            traindir = os.path.join(self.root_dir+'/'+'imagenet/train')
            valdir   = os.path.join(self.root_dir+'/'+'imagenet/val')
            train_dataset = datasets.ImageFolder(traindir,transform_train)
            test_dataset = datasets.ImageFolder(valdir,transform_test)
        elif self.dataset == 'miniimagenet':
            train_dir = os.path.join(self.root_dir+'/'+'miniimagenet/train')
            test_dir = os.path.join(self.root_dir+'/'+'miniimagenet/test')
            train_dataset = datasets.ImageFolder(train_dir,transform=transform_train)
            test_dataset = datasets.ImageFolder(test_dir,transform=transform_test)
        elif self.dataset == 'tinyimagenet':
            # Expect directory layout from the official Tiny-ImageNet:
            # {root_dir}/tiny-imagenet-200/train/
            # {root_dir}/tiny-imagenet-200/val/images
            # {root_dir}/tiny-imagenet-200/val/val_annotations.txt
            base_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
            train_dir = os.path.join(base_dir, 'train')
            val_img_dir = os.path.join(base_dir, 'val', 'images')
            val_anno = os.path.join(base_dir, 'val', 'val_annotations.txt')
            # train: standard ImageFolder (200 wnids -> indices)
            train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
            class_to_idx = train_dataset.class_to_idx  # wnid->index mapping
            # val: parse val_annotations.txt to map filename -> wnid, then to index
            annotations_map = _load_tinyimagenet_val_annotations(val_anno)
            test_dataset = TinyImageNetValDataset(
                annotations_map=annotations_map,
                img_dir=val_img_dir,
                class_to_idx=class_to_idx,
                transform=transform_test
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.num_workers,
                                                  pin_memory=True)
        return train_loader,test_loader