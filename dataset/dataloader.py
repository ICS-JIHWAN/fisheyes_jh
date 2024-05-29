import os
import torch.utils.data as data
import torchvision.transforms as transforms

from random import shuffle
from PIL import Image

STATS = {
    "vit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


class Dataset(data.Dataset):
    def __init__(self, image_dir, image_size, norm=None, direction='s2t'):
        super(Dataset, self).__init__()
        #
        self.direction = direction  # s2t or t2s
        #
        self.s_path = os.path.join(image_dir, "train_source_image")  # Source image [Flatten]
        self.t_path = os.path.join(image_dir, "train_target_image")  # Target image [Fisheye]
        self.source_filenames = [x for x in os.listdir(self.s_path)]  # s 폴더에 있는 파일 목록
        self.target_filenames = [x for x in os.listdir(self.t_path)]  # t 폴더에 있는 파일 목록
        #
        # Normalization value : mean & std
        if norm is not None:
            mean = STATS[norm]['mean']
            std = STATS[norm]['std']
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        #
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # 이미지 크기 조정
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
        a = Image.open(os.path.join(self.s_path, self.source_filenames[index % self.source_len])).convert('RGB')  # Source image
        b = Image.open(os.path.join(self.t_path, self.target_filenames[index % self.target_len])).convert('RGB')  # Target image

        # Image pre-processing
        a = self.transform(a)
        b = self.transform(b)

        # Shuffle target list at the last batch
        if self.count_shuffle == self.__len__():
            shuffle(self.target_filenames)  # Update target list
            self.count_shuffle = 0

        # Return data according to direction
        if self.direction == "s2t":  # Flatten -> Fisheye
            return a, b
        else:  # Fisheye -> Flatten
            return b, a

    def __len__(self):
        return max(self.source_len, self.target_len)
