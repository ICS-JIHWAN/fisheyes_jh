import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from random import shuffle
from PIL import Image

IGNOR_INDEX = 255
STATS = {
    "vit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}
SEG_COLOR = {
    0: (0, 0, 0),  # Road
    1: (255, 255, 0),  # Sidewalk
    2: (102, 102, 0),  # Construction
    3: (255, 0, 255),  # Fence
    4: (102, 0, 102),  # Pole
    5: (0, 255, 0),  # Traffic light
    6: (0, 0, 255),  # Traffic sign
    7: (102, 255, 102),  # Nature
    8: (204, 255, 255),  # Sky
    9: (255, 0, 0),  # Person
    10: (0, 102, 51),  # Rider
    11: (153, 0, 255),  # Car
    12: (254, 254, 254)  # Background
}


class Dataset(data.Dataset):
    def __init__(self, image_dir, image_size, n_cls, norm=None, direction='s2t'):
        super(Dataset, self).__init__()
        #
        self.n_cls = n_cls
        self.direction = direction  # s2t or t2s
        self.img_size = image_size
        #
        self.s_path = os.path.join(image_dir, "train_source_image")  # Source image [Flatten]
        self.t_path = os.path.join(image_dir, "train_target_image")  # Target image [Fisheye]
        self.g_path = os.path.join(image_dir, "train_source_gt")
        self.source_filenames = [x for x in os.listdir(self.s_path)]  # s 폴더에 있는 파일 목록
        self.target_filenames = [x for x in os.listdir(self.t_path)]  # t 폴더에 있는 파일 목록
        self.gt_filenames = [x for x in os.listdir(self.g_path)]
        #
        # Normalization value : mean & std
        if norm is not None:
            mean = STATS[norm]['mean']
            std = STATS[norm]['std']
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        #
        # Preprocessing
        self.gaussian_mask = cv2.getGaussianKernel(5, 3)
        self.sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Numpy -> Tensor
                transforms.Normalize(mean=mean, std=std)  # Normalization : -1 ~ 1 range
            ])
        #
        self.source_len = len(self.source_filenames)
        self.target_len = len(self.target_filenames)
        #
        self.count_shuffle = 0

    def __getitem__(self, index):
        self.count_shuffle += 1
        # Source & Target image load
        a = cv2.imread(os.path.join(self.s_path, self.source_filenames[index % self.source_len]), cv2.IMREAD_COLOR)
        b = cv2.imread(os.path.join(self.t_path, self.target_filenames[index % self.target_len]), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(self.g_path, self.gt_filenames[index % self.source_len]), cv2.IMREAD_GRAYSCALE)

        # Image pre-processing
        a = cv2.resize(a, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        a = self.transforms(a)
        b = self.transforms(b)
        #
        gt, ignore_mask = self.label_to_one_hot_label(torch.from_numpy(np.expand_dims(gt, axis=0).copy()), self.n_cls)

        # Shuffle target list at the last batch
        if self.count_shuffle == self.__len__():
            shuffle(self.target_filenames)  # Update target list
            self.count_shuffle = 0

        # Return data according to direction
        if self.direction == "s2t":  # Flatten -> Fisheye
            return a, b, gt, ignore_mask
        else:  # Fisheye -> Flatten
            return b, a, gt, ignore_mask

    def __len__(self):
        return max(self.source_len, self.target_len)

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
        #
        ignore_mask = (gt != IGNOR_INDEX)
        #
        gt[torch.where(gt == IGNOR_INDEX)] = n_cls
        gt_map = torch.nn.functional.one_hot(gt.type(torch.int64), n_cls + 1).transpose(0, 3).squeeze(-1)
        #
        gt_map = gt_map + eps
        #
        return gt_map, ignore_mask
