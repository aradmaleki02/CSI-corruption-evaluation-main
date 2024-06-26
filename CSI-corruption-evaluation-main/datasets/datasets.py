import os
import pickle
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'

CIFAR10_SUPERCLASS = list(range(10))  # one class
CIFAR100_CORUPTION_SUPERCLASS = list(range(20))  # one class

IMAGENET_SUPERCLASS = list(range(30))  # one class

IMAGENET30_SUPERCLASS = list(range(2))

FMNIST_SUPERCLASS = list(range(2))

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


def sparse2coarse(targets):
    coarse_labels = np.array(
        [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3,
         14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5,
         10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
         12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
         13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
         17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12,
         1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
         15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
         19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13, ])
    return coarse_labels[targets]


class CIFAR_CORRUCPION(Dataset):
    def __init__(self, transform=None, normal_idx=[0], cifar_corruption_label='CIFAR-10-C/labels.npy',
                 cifar_corruption_data='./CIFAR-10-C/defocus_blur.npy'):
        self.labels_10 = np.load(cifar_corruption_label)
        self.labels_10 = self.labels_10[40000:50000]
        if cifar_corruption_label == 'CIFAR-100-C/labels.npy':
            self.labels_10 = sparse2coarse(self.labels_10)
        self.data = np.load(cifar_corruption_data)
        self.data = self.data[40000:50000]
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels_10[index]
        if self.transform:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


class MNIST_Dataset(Dataset):
    def __init__(self, train, test_id=1, transform=None):
        self.transform = transform
        self.train = train
        self.test_id = test_id
        if train:
            with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/train_normal.pkl',
                      'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.labels = [0] * len(self.images)
        else:
            if test_id == 1:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_main.pkl',
                          'rb') as f:
                    normal_test = pickle.load(f)
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_main.pkl',
                          'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])
            else:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_shifted.pkl',
                          'rb') as f:
                    normal_test = pickle.load(f)
                with open(
                        '/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_shifted.pkl',
                        'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])

        if self.transform is not None:
            image = self.transform(image)

        height = image.shape[1]
        width = image.shape[2]
        target = 0 if self.train else self.labels[index]

        return image, target

    def __len__(self):
        return len(self.images)


class FMNIST_Dataset(Dataset):
    def __init__(self, train, test_id=1, transform=None):
        self.transform = transform
        self.train = train
        self.test_id = test_id
        if train:
            with open('/kaggle/input/diagvib-6-fmnist-dataset/content/fmnist_shifted_dataset/train_normal.pkl',
                      'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.labels = [0] * len(self.images)
        else:
            if test_id == 1:
                with open('/kaggle/input/diagvib-6-fmnist-dataset/content/fmnist_shifted_dataset/test_normal_main.pkl',
                          'rb') as f:
                    normal_test = pickle.load(f)
                with open(
                        '/kaggle/input/diagvib-6-fmnist-dataset/content/fmnist_shifted_dataset/test_abnormal_main.pkl',
                        'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])
            else:
                with open(
                        '/kaggle/input/diagvib-6-fmnist-dataset/content/fmnist_shifted_dataset/test_normal_shifted.pkl',
                        'rb') as f:
                    normal_test = pickle.load(f)
                with open(
                        '/kaggle/input/diagvib-6-fmnist-dataset/content/fmnist_shifted_dataset/test_abnormal_shifted.pkl',
                        'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])

        if self.transform is not None:
            image = self.transform(image)
        target = 0 if self.train else self.labels[index]

        return image, target

    def __len__(self):
        return len(self.images)


import random


class ISIC2018(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class GTA(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class GTA_Test(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


def get_cityscape_globs():
    from glob import glob
    import random
    normal_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/ID/*')
    anomaly_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/OOD/*')

    random.seed(42)
    random.shuffle(normal_path)
    train_ratio = 0.7
    separator = int(train_ratio * len(normal_path))
    normal_path_train = normal_path[:separator]
    normal_path_test = normal_path[separator:]

    return normal_path_train, normal_path_test, anomaly_path


def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]

    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood


class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all',
                 count=-1, return_num=2):
        self.transform = transform
        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []
        self.return_num = return_num

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        target = 0 if self.train else self.labels[idx]

        return img, target

def prepare_br35h_dataset_files():
    normal_path35 = '/kaggle/input/brain-tumor-detection/no'
    anomaly_path35 = '/kaggle/input/brain-tumor-detection/yes'

    print(f"len(os.listdir(normal_path35)): {len(os.listdir(normal_path35))}")
    print(f"len(os.listdir(anomaly_path35)): {len(os.listdir(anomaly_path35))}")

    print('cnt')

    Path('./Br35H/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/anomaly', f))

    anom35 = os.listdir(anomaly_path35)
    for f in anom35:
        shutil.copy2(os.path.join(anomaly_path35, f), './Br35H/dataset/test/anomaly')


    normal35 = os.listdir(normal_path35)
    random.shuffle(normal35)
    ratio = 0.7
    sep = round(len(normal35) * ratio)

    Path('./Br35H/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./Br35H/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/normal', f))

    flist = [f for f in os.listdir('./Br35H/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/train/normal', f))

    for f in normal35[:sep]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/train/normal')
    for f in normal35[sep:]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/test/normal')


def prepare_brats2015_dataset_files():
    import pandas as pd
    labels = pd.read_csv('/kaggle/input/brain-tumor/Brain Tumor.csv')
    labels = labels[['Image', 'Class']]
    labels.tail() # 0: no tumor, 1: tumor

    labels.head()

    brats_path = '/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor'
    lbl = dict(zip(labels.Image, labels.Class))

    keys = lbl.keys()
    normalbrats = [x for x in keys if lbl[x] == 0]
    anomalybrats = [x for x in keys if lbl[x] == 1]

    Path('./brats/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./brats/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/anomaly', f))

    flist = [f for f in os.listdir('./brats/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/normal', f))

    flist = [f for f in os.listdir('./brats/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/train/normal', f))

    ratio = 0.7
    random.shuffle(normalbrats)
    bratsep = round(len(normalbrats) * ratio)

    for f in anomalybrats:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/anomaly')
    for f in normalbrats[:bratsep]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/train/normal')
    for f in normalbrats[bratsep:]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/normal')


class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)

        has_anomaly = 0 if self.test_label[idx] == 0 else 1

        return image, has_anomaly


class BrainTrain(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image_paths = glob('./Br35H/dataset/train/normal/*')
        brats_mod = glob('./brats/dataset/train/normal/*')
        random.seed(1)
        random_brats_images = random.sample(brats_mod, 50)
        self.image_paths.extend(random_brats_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image, 0


def three_digits(a: int):
    x = str(a)
    if len(x) == 1:
        return f'00{a}'
    if len(x) == 2:
        return f'0{a}'
    return x

import pandas as pd
class WBCDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2,
                 labels1: pd.DataFrame, labels2: pd.DataFrame, transform=None, train=True, test_id=1, ratio=0.7):
        self.transform = transform
        self.root1 = root1
        self.root2 = root2
        self.labels1 = labels1
        self.labels2 = labels2
        self.train = train
        self.test_id = test_id
        self.targets = []
        labels1 = labels1[labels1['class label'] != 5]
        labels2 = labels2[labels2['class'] != 5]

        normal_df = labels1[labels1['class label'] == 1]
        self.normal_paths = [os.path.join(root1, f'{three_digits(x)}.bmp') for x in list(normal_df['image ID'])]
        random.seed(42)
        random.shuffle(self.normal_paths)
        self.separator = int(ratio * len(self.normal_paths))
        self.train_paths = self.normal_paths[:self.separator]

        if self.train:
            self.image_paths = self.train_paths
            self.targets = [0] * len(self.image_paths)
        else:
            if self.test_id == 1:
                all_images = glob(os.path.join(root1, '*.bmp'))
                self.image_paths = [x for x in all_images if x not in self.train_paths]
                self.image_paths = [x for x in self.image_paths if
                                    int(os.path.basename(x).split('.')[0]) in labels1['image ID'].values]
                ids = [os.path.basename(x).split('.')[0] for x in self.image_paths]
                ids_labels = list(labels1[labels1['image ID'] == int(x)]['class label'] for x in ids)
                self.targets = [0 if x.item() == 1 else 1 for x in ids_labels]
            else:
                self.image_paths = glob(os.path.join(root2, '*.bmp'))
                self.image_paths = [x for x in self.image_paths if int(os.path.basename(x).split('.')[0])
                                    in labels2['image ID'].values]
                self.targets = [
                    0 if labels2[labels2['image ID'] == int(os.path.basename(x).split('.')[0])]['class'].item() == 1
                    else 1 for x in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        height = img.shape[1]
        width = img.shape[2]
        target = 0 if self.train else self.targets[idx]

        return img, target



def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    download = True
    image_size = (P.image_size, P.image_size, 3)
    print(image_size)
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    elif dataset == 'fmnist':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif dataset == 'mn':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif dataset in ['isic', 'gta', 'waterbirds', 'brain', 'wbc']:
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'fmnist':
        image_size = (224, 224, 3)
        n_classes = 2
        test_set = FMNIST_Dataset(train=False, transform=test_transform, test_id=1)
        test_set2 = FMNIST_Dataset(train=False, transform=test_transform, test_id=2)
        if P.test_id == 2:
            test_set = test_set2
        train_set = FMNIST_Dataset(train=True, transform=train_transform)
    elif dataset == 'mn':
        image_size = (224, 224, 3)
        n_classes = 2
        test_set = MNIST_Dataset(train=False, transform=test_transform, test_id=1)
        test_set2 = MNIST_Dataset(train=False, transform=test_transform, test_id=2)
        if P.test_id == 2:
            test_set = test_set2
        train_set = MNIST_Dataset(train=True, transform=train_transform)
    elif dataset == 'wbc':
        n_classes = 2
        import pandas as pd
        df1 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 1.csv')
        df2 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 2.csv')
        test_set = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1',
                               '/kaggle/working/segmentation_WBC/Dataset 2',
                               df1, df2, transform=train_transform, train=False, test_id=1)
        if P.test_id == 2:
            test_set = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1',
                               '/kaggle/working/segmentation_WBC/Dataset 2',
                               df1, df2, transform=test_transform, train=False, test_id=2)
        train_set = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1',
                               '/kaggle/working/segmentation_WBC/Dataset 2',
                               df1, df2, transform=test_transform, train=True)
    elif dataset == 'brain':
        if P.brain_prepared == 0:
            prepare_br35h_dataset_files()
            prepare_brats2015_dataset_files()
            P.brain_prepared = 1
        n_classes = 2
        train_set = BrainTrain(transform=train_transform)
        test_set = BrainTest(transform=test_transform, test_id=1)
        if P.test_id == 2:
            test_set = BrainTest(transform=test_transform, test_id=2)
    elif dataset == 'waterbirds':
        n_classes = 2
        import pandas as pd
        df = pd.read_csv('/kaggle/input/waterbird/waterbird/metadata.csv')
        train_set = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=train_transform, train=True, count_train_landbg=3500,
                                       count_train_waterbg=100)
        test_set = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=test_transform, train=False, count_train_landbg=3500,
                                       count_train_waterbg=100, mode='bg_land')
        if P.test_id == 2:
            test_set = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=test_transform, train=False, count_train_landbg=3500,
                                       count_train_waterbg=100, mode='bg_water')
    elif dataset == 'isic':
        # image_size = (32, 32, 3)
        n_classes = 2
        from glob import glob
        import pandas as pd
        train_path = glob('/kaggle/input/isic-task3-dataset/dataset/train/NORMAL/*')
        train_label = [0] * len(train_path)
        test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')
        test_anomaly_label = [1] * len(test_anomaly_path)
        test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')
        test_normal_label = [0] * len(test_normal_path)

        test_label = test_anomaly_label + test_normal_label
        test_path = test_anomaly_path + test_normal_path

        df = pd.read_csv('/kaggle/input/pad-ufes-20/PAD-UFES-20/metadata.csv')

        shifted_test_label = df["diagnostic"].to_numpy()
        shifted_test_label = (shifted_test_label != "NEV")
        # shifted_test_label = [0 if shifted_test_label[i] is False else 1 for i in range(len(shifted_test_label))]

        shifted_test_path = df["img_id"].to_numpy()
        shifted_test_path = '/kaggle/input/pad-ufes-20/PAD-UFES-20/Dataset/' + shifted_test_path

        train_set = ISIC2018(image_path=train_path, labels=train_label, transform=train_transform)
        test_set = ISIC2018(image_path=test_path, labels=test_label, transform=test_transform)
        if P.test_id == 2:
            test_set = ISIC2018(image_path=shifted_test_path, labels=shifted_test_label, transform=test_transform)
    elif dataset == 'gta':
        n_classes = 2
        normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
        test_path = normal_path_test + anomaly_path
        test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
        train_label = [0] * len(normal_path_train)
        glob_train_id, glob_test_id, glob_ood = get_gta_globs()
        train_set = GTA(image_path=normal_path_train, labels=train_label,
                        transform=train_transform)
        test_set = GTA_Test(image_path=test_path, labels=test_label,
                            transform=test_transform)
        if P.test_id == 2:
            test_set = GTA_Test(image_path=glob_test_id + glob_ood,
                                labels=[0] * len(glob_test_id) + [1] * len(glob_ood),
                                transform=test_transform)

    elif dataset == 'svhn':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
    elif dataset == 'svhn-10':
        image_size = (32, 32, 3)
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'svhn-10-corruption':
        image_size = (32, 32, 3)

        def gaussian_noise(image, mean=P.noise_mean, std=P.noise_std, noise_scale=P.noise_scale):
            image = image + (torch.randn(image.size()) * std + mean) * noise_scale
            return image

        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Lambda(gaussian_noise)
        ])

        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'mnist':
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'imagenet30':
        n_classes = 2
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        anomaly_testset = datasets.ImageFolder('./one_class_test', transform=transform)
        for i in range(len(anomaly_testset)):
            anomaly_testset.targets[i] = 1
        anomaly_trainset = datasets.ImageFolder('./one_class_train', transform=transform)
        for i in range(len(anomaly_trainset)):
            anomaly_trainset.targets[i] = 1
        test_set = anomaly_testset
        train_set = anomaly_trainset
    elif dataset == 'fashion-mnist':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.FashionMNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'cifar10-corruption':
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'cifar100-corruption':
        n_classes = 100
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_label='CIFAR-100-C/labels.npy',
                                    cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=transform)

        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_pil' or dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_pil' or dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10' or dataset == 'cifar10-corruption' or dataset == 'svhn' or dataset == 'svhn-10-corruption' or dataset == 'svhn-10' or dataset == 'fashion-mnist' or dataset == 'mnist':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet30':
        return IMAGENET30_SUPERCLASS
    elif dataset == "cifar100-corruption":
        return CIFAR100_CORUPTION_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        return FMNIST_SUPERCLASS


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    try:
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        # SVHN
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
