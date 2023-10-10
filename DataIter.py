import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.dataset import Dataset
import random
import os
import time
import cv2
from scipy.ndimage.interpolation import rotate
import scipy.ndimage
import mindspore.ops.operations as P
import mindspore.nn as nn

class MBDataIterSensi_Resis(Dataset):
    def __init__(self, data_file, data_dir, phase="train", crop_size=48, crop_depth=16, sample_size=64, aug=1, sample_phase='over'):
        self.phase = phase
        self.data_arr = np.load(data_file, allow_pickle=True).tolist()
        self.data_dir = data_dir
        sensitivity_lst = []  # mal
        resistant_lst = []  # ben

        for i in range(len(self.data_arr)):
            if 'ds' in self.data_arr[i]['path']:
                sensitivity_lst.append(self.data_arr[i])
            else:
                resistant_lst.append(self.data_arr[i])
        print(len(sensitivity_lst), len(resistant_lst))

        if phase == "train":
            minus_ben = len(resistant_lst) - len(sensitivity_lst)
            if sample_phase == 'over':
                random.shuffle(sensitivity_lst)
                if minus_ben > len(sensitivity_lst):
                    minus_ben = minus_ben - len(sensitivity_lst)
                    mal_cop = sensitivity_lst[:minus_ben] + sensitivity_lst
                else:
                    mal_cop = sensitivity_lst[:minus_ben]
                self.data_lst = mal_cop * aug + sensitivity_lst * aug + resistant_lst * aug

            elif sample_phase == 'under':
                random.shuffle(resistant_lst)
                ben_cop = resistant_lst[:len(sensitivity_lst)]
                self.data_lst = ben_cop + sensitivity_lst
            else:
                random.shuffle(resistant_lst)
                random.shuffle(sensitivity_lst)
                self.data_lst = resistant_lst * aug + sensitivity_lst * aug
        else:
            self.data_lst = resistant_lst + sensitivity_lst

        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size, zslice=crop_depth, phase=self.phase)
        self.augm = Augmentation(phase=self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        cur_dir = self.data_lst[idx]['path'].split('@')[0]
        label = np.zeros((13,), dtype=np.float32)
        if 'ds' in self.data_lst[idx]['path']:
            label[0] = 0.0
        else:
            label[0] = 1.0

        label[1:] = self.data_lst[idx]['character']

        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx
        imgs = self.crop(cur_dir)

        if self.phase == "train":
            imgs = self.augm(imgs)

        imgs = imgs[np.newaxis, :, :, :]
        return Tensor(imgs.astype(np.float32)), Tensor(label.astype(np.float32)), cur_dir

    def __len__(self):
        return len(self.data_lst)


class Crop:
    def __init__(self, size=48, zslice=16, phase='train'):
        self.crop_size = size
        self.zslice = zslice
        self.phase = phase
        self.random_crop = RandomCenterCrop(size, zslice)
        self.center_crop = CenterCrop(size, zslice)

    def normalize(self, img):
        MIN_BOUND = -1300
        MAX_BOUND = 500
        img[img > MAX_BOUND] = MAX_BOUND
        img[img < MIN_BOUND] = MIN_BOUND
        img = img.astype(np.float32)
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        return img

    def __call__(self, img_npy):
        img = np.load(img_npy)
        if self.phase == "test":
            img_r = self.center_crop(img)
        else:
            img_r = self.random_crop(img)

        img_r = self.normalize(img_r)
        for shapa_ in img_r.shape[1:]:
            if shapa_ not in [16, 32, 48, 64, 96, 112]:
                print(shapa_)
        return img_r


class RandomCenterCrop:
    def __init__(self, size, zslice):
        assert size in [16, 32, 48, 64, 96, 112] and zslice in [6, 8, 10, 16, 32, 48, 64]
        self.size = (int(size), int(size))
        self.zslice = zslice
        if size == 16:
            self.randseed = 4
        elif size == 32:
            self.randseed = 6
        elif size == 48:
            self.randseed = 8
        elif size == 64:
            self.randseed = 10
        elif size == 96:
            self.randseed = 12
        elif size == 112:
            self.randseed = 14

    def __call__(self, data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice

        i = random.randint(-self.randseed, self.randseed)
        j = random.randint(-self.randseed, self.randseed)

        x_start = max(int(round((x - des_w) / 2.) + i), 0)
        x_end = min(x_start + des_w, x)

        y_start = max(int(round((y - des_h) / 2.) + j), 0)
        y_end = min(y_start + des_h, y)

        s_start = max(int(round((s - des_s) / 2.)), 0)
        s_end = min(s_start + des_s, s)

        data = data[s_start: s_start + des_s,
                    y_start: y_start + des_h,
                    x_start: x_start + des_w]

        pad_size = (des_s - (s_end - s_start), des_h - (y_end - y_start), des_w - (x_end - x_start))
        pad_edge = (
            (int(pad_size[0] / 2), pad_size[0] - int(pad_size[0] / 2)),
            (int(pad_size[1] / 2), pad_size[1] - int(pad_size[1] / 2)),
            (int(pad_size[2] / 2), pad_size[2] - int(pad_size[2] / 2))
        )

        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')

        data = data.reshape(des_s, des_h, des_w)
        return data


class CenterCrop:
    def __init__(self, size, zslice):
        assert size in [16, 32, 48, 64, 96, 112] and zslice in [6, 8, 10, 16, 32, 48, 64]
        self.size = (int(size), int(size))
        self.zslice = zslice

    def __call__(self, data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice
        x_start = max(int(round((x - des_w) / 2.)), 0)
        x_end = min(x_start + des_w, x)

        y_start = max(int(round((y - des_h) / 2.)), 0)
        y_end = min(y_start + des_h, y)

        s_start = max(int(round((s - des_s) / 2.)), 0)
        s_end = min(s_start + des_s, s)

        data = data[s_start: s_end,
                    y_start: y_end,
                    x_start: x_end]

        pad_size = (des_s - (s_end - s_start), des_h - (y_end - y_start), des_w - (x_end - x_start))
        pad_edge = (
            (int(pad_size[0] / 2), pad_size[0] - int(pad_size[0] / 2)),
            (int(pad_size[1] / 2), pad_size[1] - int(pad_size[1] / 2)),
            (int(pad_size[2] / 2), pad_size[2] - int(pad_size[2] / 2))
        )

        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')

        try:
            data = data.reshape(des_s, des_h, des_w)
        except:
            import pdb;
            pdb.set_trace()
        return data


class RandomCenterCrop:
    def __init__(self, size, zslice):
        assert size in [16, 32, 48, 64, 96, 112] and zslice in [6, 8, 10, 16, 32, 48, 64]
        self.size = (int(size), int(size))
        self.zslice = zslice
        if size == 16:
            self.randseed = 4
        elif size == 32:
            self.randseed = 6
        elif size == 48:
            self.randseed = 8
        elif size == 64:
            self.randseed = 10
        elif size == 96:
            self.randseed = 12
        elif size == 112:
            self.randseed = 14

    def __call__(self, data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice

        i = random.randint(-self.randseed, self.randseed)
        j = random.randint(-self.randseed, self.randseed)

        x_start = max(int(round((x - des_w) / 2.) + i), 0)
        x_end = min(x_start + des_w, x)

        y_start = max(int(round((y - des_h) / 2.) + j), 0)
        y_end = min(y_start + des_h, y)

        s_start = max(int(round((s - des_s) / 2.)), 0)
        s_end = min(s_start + des_s, s)

        data = data[s_start: s_start + des_s,
                    y_start: y_start + des_h,
                    x_start: x_start + des_w]

        pad_size = (des_s - (s_end - s_start), des_h - (y_end - y_start), des_w - (x_end - x_start))
        pad_edge = (
            (int(pad_size[0] / 2), pad_size[0] - int(pad_size[0] / 2)),
            (int(pad_size[1] / 2), pad_size[1] - int(pad_size[1] / 2)),
            (int(pad_size[2] / 2), pad_size[2] - int(pad_size[2] / 2))
        )

        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')

        data = data.reshape(des_s, des_h, des_w)
        return data


class Augmentation:
    def __init__(self, phase='train'):
        self.phase = phase

    def __call__(self, img_r):
        if self.phase == "train":
            ran_type = random.randint(0, 1)
            if ran_type:
                angle1 = np.random.rand() * 180
                img_r = rotate(img_r, angle1, axes=(1, 2), reshape=False, mode='nearest')

            ran_type = random.randint(0, 1)
            if ran_type:
                img_r = cv2.flip(img_r, 0)

            ran_type = random.randint(0, 1)
            if ran_type:
                img_r = cv2.flip(img_r, 1)

            ran_type = random.randint(0, 1)
            if ran_type:
                img_r = np.flip(img_r, 2)

        return img_r


class MBDataIterResisClassify(Dataset):
    def __init__(self, data_file, data_dir, phase="train", crop_size=48, crop_depth=16, sample_size=224, aug=1, sample_phase='over'):
        self.data_dir = data_dir
        self.phase = phase
        self.data_arr = np.load(data_file, allow_pickle=True).tolist()
        rifr_lst = []
        mdr_lst = []
        xdr_lst = []

        for i in range(len(self.data_arr)):
            if 'rifr' in self.data_arr[i]['path']:
                rifr_lst.append(self.data_arr[i])
            elif 'mdr' in self.data_arr[i]['path']:
                mdr_lst.append(self.data_arr[i])
            elif 'xdr' in self.data_arr[i]['path']:
                xdr_lst.append(self.data_arr[i])
        print(len(rifr_lst), len(mdr_lst), len(xdr_lst))
        if phase == "train":
            self.data_lst = rifr_lst * aug + mdr_lst * aug * 2 + xdr_lst * aug
        else:
            self.data_lst = rifr_lst + mdr_lst + xdr_lst

        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size, zslice=crop_depth, phase=self.phase)
        self.augm = Augmentation(phase=self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        cur_dir = self.data_lst[idx]["path"].split('@')[0]
        label = np.zeros((13,), dtype=np.float32)
        if 'rifr' in self.data_lst[idx]["path"]:
            label[0] = 0.0
        elif 'mdr' in self.data_lst[idx]["path"]:
            label[0] = 1.0
        elif 'xdr' in self.data_lst[idx]["path"]:
            label[0] = 2.0
        label[1:] = self.data_lst[idx]['character']

        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx
        imgs = self.crop(cur_dir)

        if self.phase == "train":
            imgs = self.augm(imgs)

        imgs = imgs[np.newaxis, :, :, :]
        return Tensor(imgs.astype(np.float32)), Tensor(label.astype(np.float32)), cur_dir

    def __len__(self):
        return len(self.data_lst)
