# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import numpy as np
import torchvision.transforms as transforms
import data.cvtransforms as transforms_dct
from .dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod
from data import train_upscaled_static_mean, train_upscaled_static_std
from data import train_upscaled_static_dct_direct_mean, train_upscaled_static_dct_direct_std
from data import train_dct_subset_mean, train_dct_subset_std
from data import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, train_cr_mean_resized, train_cr_std_resized
from data import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, train_cr_mean_upscaled, train_cr_std_upscaled

from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split


class TransformLoader:
    def __init__(self, image_size, normalize_param=None, jitter_param=None):
        self.image_size = image_size
        self.rotation_param = dict(degrees=20)
        self.random_erasing_param = dict(
            p=0.5,
            scale=(0.02, 0.1),
            value=1.0,
            inplace=False,
        )
        self.random_adjust_sharpness_param = dict(
            p=0.2,
            sharpness_factor=2,
        )
        if normalize_param is not None:
            self.normalize_param = normalize_param
        else:
            self.normalize_param = dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        if jitter_param is not None:
            self.jitter_param = jitter_param
        else:
            self.jitter_param = dict(
                Brightness=0.4,
                Contrast=0.4,
                Color=0.4,
            )

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = transforms_dct.ImageJitter(self.jitter_param)
            return method
        if transform_type == 'Rescale':
            method = transforms_dct.Rescale()
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type == 'CenterCrop':
            return method(self.image_size) 
        elif transform_type == 'Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(**self.rotation_param)
        elif transform_type == 'RandomErasing':
            return method(**self.random_erasing_param)
        elif transform_type == 'RandomAdjustSharpness':
            return method(**self.random_adjust_sharpness_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = [
                'RandomResizedCrop',
                'ImageJitter',
                'RandomHorizontalFlip',
                # 'RandomRotation',
                # 'RandomAdjustSharpness',
                'ToTensor',
                'Normalize',
                # 'Rescale',
                # 'RandomErasing',
            ]
        else:
            transform_list = [
                'Resize',
                'CenterCrop',
                'ToTensor',
                'Normalize',
                # 'Rescale',
            ]
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_composed_transform_dct_np(self, aug=False, filter_size=8, channels=24, dct_pattern='default'):
        """ Assuming input is a numpy array (eg. custom datasets)"""
        if aug:
            transform = transforms_dct.Compose([
                transforms_dct.RandomResizedCrop(filter_size * self.image_size),
                transforms_dct.ImageJitter(self.jitter_param),
                transforms_dct.RandomHorizontalFlip(),
                # transforms.RandomRotation(**self.rotation_param),
                # transforms.RandomAdjustSharpness(**self.random_adjust_sharpness_param),
                transforms_dct.GetDCT(filter_size),
                transforms_dct.UpScaleDCT(size=self.image_size),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(
                    channels=channels,
                    pattern=dct_pattern,
                ),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                    # train_y_mean_resized,  train_y_std_resized,
                    # train_cb_mean_resized, train_cb_std_resized,
                    # train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=channels
                ),
                # transforms.RandomErasing(**self.random_erasing_param),
                # transforms_dct.Rescale()
                lambda x: (x[0]),
            ])
        else:
            transform = transforms_dct.Compose([
                transforms_dct.Resize(int(filter_size * self.image_size * 1.15)),
                transforms_dct.CenterCrop(filter_size * self.image_size),
                transforms_dct.GetDCT(filter_size),
                transforms_dct.UpScaleDCT(size=self.image_size),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(
                    channels=channels,
                    pattern=dct_pattern,
                ),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                    # train_y_mean_resized,  train_y_std_resized,
                    # train_cb_mean_resized, train_cb_std_resized,
                    # train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=channels,
                ),
                # transforms_dct.Rescale()
                lambda x: (x[0]),
            ])
        return transform

    def get_composed_transform_dct_img(self, aug=False, filter_size=8, channels=24, dct_pattern='default'):
        """ Assuming input is PIL image (eg. when using torch.datasets) """
        if aug:
            transform = transforms_dct.Compose([
                # transforms.RandomRotation(**self.rotation_param),
                # transforms.RandomAdjustSharpness(**self.random_adjust_sharpness_param),
                lambda x: (np.array(x)),
                # transforms.ToTensor(),
                # lambda x: (x * 255).numpy().transpose(1, 2, 0).astype(np.uint8),
                transforms_dct.RandomResizedCrop(filter_size * self.image_size),
                transforms_dct.ImageJitter(self.jitter_param),
                transforms_dct.RandomHorizontalFlip(),
                # transforms.RandomRotation(**self.rotation_param),
                # transforms.RandomAdjustSharpness(**self.random_adjust_sharpness_param),
                # transforms.RandomResizedCrop(filter_size * upscale_spatial),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4,
                #     hue=0.4,
                # ),
                transforms_dct.GetDCT(filter_size),
                transforms_dct.UpScaleDCT(size=self.image_size),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(
                    channels=channels,
                    pattern=dct_pattern,
                ),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                    # train_y_mean_resized,  train_y_std_resized,
                    # train_cb_mean_resized, train_cb_std_resized,
                    # train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=channels
                ),
                # transforms.RandomErasing(**self.random_erasing_param),
                # transforms_dct.Rescale()
                lambda x: (x[0]),
            ])
        else:
            transform = transforms_dct.Compose([
                lambda x: (np.array(x)),
                # transforms.ToTensor(),
                # lambda x: (x * 255).numpy().transpose(1, 2, 0).astype(np.uint8),
                # lambda x: cv2.cvtColor((x * 255).numpy().transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_BGR2RGB),
                transforms_dct.Resize(int(filter_size * self.image_size * 1.15)),
                transforms_dct.CenterCrop(filter_size * self.image_size),
                transforms_dct.GetDCT(filter_size),
                transforms_dct.UpScaleDCT(size=self.image_size),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(
                    channels=channels,
                    pattern=dct_pattern,
                ),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                    # train_y_mean_resized,  train_y_std_resized,
                    # train_cb_mean_resized, train_cb_std_resized,
                    # train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=channels
                ),
                # transforms_dct.Rescale()
                lambda x: (x[0]),
            ])
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, normalize_param=None, jitter_param=None):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size, normalize_param, jitter_param)

    def get_data_loader(self, data_file, aug, subset=None, random_state=42): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform, dct_status=False)
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=(True if aug else False),
            num_workers=8,
            pin_memory=True,
        )
        if subset is not None:
            _, subset_idx = train_test_split(np.arange(len(dataset)), test_size=subset, random_state=random_state)
            sampler = SubsetRandomSampler(subset_idx)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                **data_loader_params,
            )
            # subset_dataset = torch.utils.data.Subset(dataset, range(subset))
            # data_loader = torch.utils.data.DataLoader(subset_dataset, **data_loader_params)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader, dataset

    def get_data_loader_dct(self, data_file, aug, filter_size, channels, subset=None, random_state=42):
        transform = self.trans_loader.get_composed_transform_dct_np(aug, filter_size, channels)
        dataset = SimpleDataset(data_file, transform, dct_status=True)
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=(True if aug else False),
            num_workers=8,
            pin_memory=True,
        )
        if subset is not None:
            _, subset_idx = train_test_split(np.arange(len(dataset)), test_size=subset, random_state=random_state)
            sampler = SubsetRandomSampler(subset_idx)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                **data_loader_params,
            )
            # subset_dataset = torch.utils.data.Subset(dataset, range(subset))
            # data_loader = torch.utils.data.DataLoader(subset_dataset, **data_loader_params)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader, dataset


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size, transform, dct_status=False)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler,  num_workers=16, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

    def get_data_loader_dct(self, data_file, aug, filter_size): 
        transform = self.trans_loader.get_composed_transform_dct(aug, filter_size)
        dataset = SetDataset(data_file, self.batch_size, transform, dct_status=True )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        data_loader_params = dict(batch_sampler=sampler,  num_workers=16, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


