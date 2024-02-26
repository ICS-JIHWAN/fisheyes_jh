import os
import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image

from data.data_agument import letterbox


class FishEyesDataset():
    def __init__(self, data_dict, hyp=None, img_size=640, augment=False, input_gray=False):
        self.root_path = data_dict['root_path']
        source_csv = os.path.join(self.root_path, 'csvfile', data_dict['train_source'])
        target_csv = os.path.join(self.root_path, 'csvfile', data_dict['train_target'])
        #
        self.__dict__.update(locals())
        #
        self.source_path_list, self.target_path_list = self.get_image_path(source_csv, target_csv)
        #
        self.len_source_data = len(self.source_path_list)
        self.len_target_data = len(self.target_path_list)

    def __len__(self):
        return max(self.len_source_data, self.len_target_data)

    def __getitem__(self, index):
        return_dict = dict()
        #
        source_path = self.source_path_list[index % self.len_source_data]
        target_path = self.target_path_list[index % self.len_target_data]
        #
        source_img = self.load_image(source_path)
        target_img = self.load_image(target_path)
        #
        if not self.input_gray:
            return_dict['source'] = np.ascontiguousarray(source_img[0].transpose((2, 0, 1))[::-1])
            return_dict['target'] = np.ascontiguousarray(target_img[0].transpose((2, 0, 1))[::-1])
        else:
            return_dict['source'] = np.ascontiguousarray(np.expand_dims(source_img[0], axis=2).transpose((2, 0, 1)))
            return_dict['target'] = np.ascontiguousarray(np.expand_dims(target_img[0], axis=2).transpose((2, 0, 1)))

        return return_dict

    def get_image_path(self, source_csv, target_csv):
        df_source = pd.read_csv(source_csv)
        df_target = pd.read_csv(target_csv)

        source_path_list = df_source.img_path.to_list()
        target_path_list = df_target.img_path.to_list()

        return source_path_list, target_path_list

    def load_image(self, path):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        try:
            if not self.input_gray:
                im = cv2.imread(os.path.join(self.root_path, path))
                assert im is not None, f"opencv cannot read image correctly or {path} not exists"
            else:
                im = cv2.imread(os.path.join(self.root_path, path), cv2.IMREAD_GRAYSCALE)
        except:
            if not self.input_gray:
                im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
                assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"
            else:
                im = np.asarray(Image.open(path).convert('L'))

        h0, w0 = im.shape[:2]  # original shape
        r = self.img_size / max(h0, w0)

        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )

        # Letterbox
        shape = (self.img_size)  # final letterboxed shape
        if self.hyp['use_letterbox']:
            if self.hyp and "letterbox_return_int" in self.hyp:
                img, ratio, pad = letterbox(im, shape, color=self.hyp['letterbox_color'], auto=False, scaleup=self.augment,
                                            return_int=self.hyp["letterbox_return_int"])
            else:
                img, ratio, pad = letterbox(im, shape, color=self.hyp['letterbox_color'], auto=False, scaleup=self.augment)
        else:
            img = cv2.resize(im, (self.img_size, self.img_size))
        return img, (h0, w0), img.shape[:2]

    def collate_fn(self, batch):
        source = list(map(lambda x: torch.from_numpy(x['source']), batch))
        target = list(map(lambda x: torch.from_numpy(x['target']), batch))

        return torch.stack(source, dim=0), torch.stack(target, dim=0)


if __name__ == '__main__':
    from utils.events import load_yaml
    from utils.config import Config

    cfg = Config.fromfile('../config/cyclegan.py')
    data_dict = load_yaml('../data/fisheyes.yaml')

    dataset_object = FishEyesDataset(
        data_dict, hyp=dict(cfg.data_aug))

    dataset_object.__getitem__(1)
