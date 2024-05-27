import os
import cv2
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from utils.events import LOGGER
from data.data_agument import barrel_distortion_np

IGNOR_INDEX = 255
STATS = {
    "vit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


class FishEyesDataset():
    def __init__(
            self,
            data_dict,
            img_size,
            n_cls,
            check_path=False,
            normalization=None,
    ):
        #
        self.root_path = Path(data_dict['root_path'])
        self.img_size = img_size
        self.normalization = normalization
        self.n_cls = n_cls
        #
        self.source_img_paths, self.source_gt_paths, self.target_img_paths = self.get_image_path(data_dict)
        #
        if check_path:
            LOGGER.info('Check matching of image paths and label paths')
            for idx in tqdm(range(len(self.source_img_paths))):
                if Path(self.source_img_paths[idx]).name != Path(self.source_gt_paths[idx]).name:
                    LOGGER.error('Miss match between image path and label path!!')
        #
        if normalization is not None:
            self.normalization = STATS[normalization].copy()
        else:
            self.normalization = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        print(f'Use normalization : {self.normalization}')
        #
        # self.gaussian_mask = cv2.getGaussianKernel(5, 3)
        self.sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.transforms = self._transform()
        #
        self.len_source_data = len(self.source_img_paths)
        self.len_target_data = len(self.target_img_paths)

    def __len__(self):
        return self.len_source_data

    def __getitem__(self, index):
        return_dict = dict()
        #
        source_img_path = os.path.join(self.root_path, self.source_img_paths[index % self.len_source_data])
        source_gt_path = os.path.join(self.root_path, self.source_gt_paths[index % self.len_source_data])
        target_img_path = os.path.join(self.root_path, self.target_img_paths[index % self.len_target_data])
        #
        source_img, source_gt, target_img = self.load_data(source_img_path, source_gt_path, target_img_path)
        #
        source_gt = self.label_to_one_hot_label(source_gt, self.n_cls)
        #
        return_dict['s_img'] = source_img[1]
        return_dict['s_crop_img'] = source_img[2]
        return_dict['s_gt'] = source_gt
        return_dict['t_img'] = target_img[1]
        return_dict['s_Osize'] = source_img[0]
        return_dict['t_Osize'] = target_img[0]
        #
        return return_dict

    def get_image_path(self, path_dict):
        #
        df_source = pd.read_csv(self.root_path / 'csvfile' / path_dict['train_source'])
        df_target = pd.read_csv(self.root_path / 'csvfile' / path_dict['train_target'])

        source_img_paths = df_source.img_path.to_list()
        source_gt_paths = df_source.gt_path.to_list()
        target_img_paths = df_target.img_path.to_list()
        #
        return source_img_paths, source_gt_paths, target_img_paths

    def load_data(self, source_img_path, source_gt_path, target_img_path):
        #
        Is = cv2.imread(source_img_path, cv2.IMREAD_COLOR)
        Isem = cv2.imread(source_gt_path, cv2.IMREAD_GRAYSCALE)
        It = cv2.imread(target_img_path, cv2.IMREAD_COLOR)
        #
        s_h0, s_w0, _ = Is.shape
        t_h0, t_w0, _ = It.shape
        #
        Is = cv2.resize(
            Is,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        Isem = cv2.resize(
            Isem,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        It = cv2.resize(
            It,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        #
        Ist_prime_crop = barrel_distortion_np(Is, self.img_size)
        #
        # Is = cv2.filter2D(Is, -1, self.gaussian_mask)  # Gaussian blurring
        Is = cv2.filter2D(Is, -1, self.sharpening_mask)  # Sharpening
        Is = self.transforms(Is)

        # It = cv2.filter2D(It, -1, self.gaussian_mask)  # Gaussian blurring
        It = cv2.filter2D(It, -1, self.sharpening_mask)  # Sharpening
        It = self.transforms(It)

        # Ist_prime_crop = cv2.filter2D(Ist_prime_crop, -1, self.gaussian_mask)
        Ist_prime_crop = cv2.filter2D(Ist_prime_crop, -1, self.sharpening_mask)
        Ist_prime_crop = self.transforms(Ist_prime_crop)
        #
        return [(s_h0, s_w0), Is, Ist_prime_crop], \
            torch.from_numpy(np.expand_dims(Isem, axis=0).copy()), \
            [(t_h0, t_w0), It]

    def label_to_one_hot_label(self, gt, n_cls=12, eps: float = 1e-6):
        # I want...
        # gt     : 1 x h x w    ex) 1 x 512 x 512
        # gt_map : c x h x w    ex) 12 x 512 x 512

        # Process of gt_map
        # one_hot(gt, IGNOR_INDEX + 1)                              shape : 1 x h x w x 256 -> 256 = background(255) + 1
        # one_hot(gt, IGNOR_INDEX + 1).transpose(0, 3)              shape : 256 x h x w x 1
        # one_hot(gt, IGNOR_INDEX + 1).transpose(0, 3).squeeze(-1)  shape : 256 x h x w
        # split(gt_map, n_cls, dim=0)[0]                            shape : 12 x h x w
        # =============================================================================================================
        # result of gt_map : 12 x h x w
        # check !!
        gt_map = torch.nn.functional.one_hot(gt.type(torch.int64), IGNOR_INDEX + 1).transpose(0, 3).squeeze(-1)

        gt_map = torch.split(gt_map, n_cls, dim=0)[0] + eps

        return gt_map

    def _transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalization['mean'], self.normalization['std']),
        ])

    def collate_fn(self, batches):
        #
        s_imgs = []
        s_crop_imgs = []
        s_gts = []
        t_imgs = []
        s_Osizes = []
        t_Osizes = []
        for batch in batches:
            s_imgs.append(batch['s_img'])
            s_crop_imgs.append(batch['s_crop_img'])
            s_gts.append(batch['s_gt'])
            t_imgs.append(batch['t_img'])
            s_Osizes.append(batch['s_Osize'])
            t_Osizes.append(batch['t_Osize'])
        #
        source_dict = {
            'img': torch.stack(s_imgs, dim=0),
            'img_crop': torch.stack(s_crop_imgs, dim=0),
            'gt': torch.stack(s_gts, dim=0),
            'ori_size': s_Osizes
        }

        target_dict = {
            'img': torch.stack(t_imgs, dim=0),
            'ori_size': t_Osizes
        }

        return source_dict, target_dict


if __name__ == '__main__':
    from utils.events import load_yaml
    from config.config_utils import Config

    cfg = Config.fromfile('../config/cfg_cyclegan.py')
    data_dict = load_yaml('../data/fisheyes.yaml')

    dataset_object = FishEyesDataset(
        data_dict, img_size=512, n_cls=12, normalization='vit')

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_object,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_object.collate_fn,
    )

    from time import time

    start = time()
    for i, data in enumerate(tqdm(train_loader)):
        print(i)
    end = time()
    print(end - start)
