import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import random

class ToTensor:
    def __call__(self, x):
        return Tensor(x.copy(), mindspore.float32)

class Resize:
    def __init__(self, size):
        self.size = size
        self.interpolate = ops.ResizeBilinear((size[0], size[1]))

    def __call__(self, x):
        return self.interpolate(x)

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        z, y, x = data.shape
        des_z, des_y, des_x = self.size
        start_z = int(round((z - des_z) / 2.))
        start_y = int(round((y - des_y) / 2.))
        start_x = int(round((x - des_x) / 2.))
        data = data[start_z: start_z + des_z,
               start_y: start_y + des_y,
               start_x: start_x + des_x]
        data = data[np.newaxis, :]
        data = data[np.newaxis, :]
        return Tensor(data, mindspore.float32)

class Normalize:
    def __init__(self, bound=[-1300.0, 500.0], cover=[0.0, 1.0]):
        self.minbound = min(bound)
        self.maxbound = max(bound)
        self.target_min = min(cover)
        self.target_max = max(cover)

    def __call__(self, x):
        out = (x - self.minbound) / (self.maxbound - self.minbound)
        out = ops.clip_by_value(out, self.target_min, self.target_max)
        return out

class TripleCenterCrop:
    def __init__(self, sizes):
        self.sizes = {}
        self.sizes['small'] = sizes[0]
        self.sizes['middle'] = sizes[1]
        self.sizes['large'] = sizes[2]
        self.totensor = ToTensor()
        self.center_crop = CenterCrop

    def __call__(self, data):
        nodule = {}
        for key, value in self.sizes.items():
            sample_size = [value[1], value[0], value[0]]
            crop = self.center_crop(sample_size)
            temp = crop(data)
            nodule[key] = self.totensor(temp)
        return nodule

class TripleRandomCrop:
    def __init__(self, sizes):
        self.sizes = {}
        self.sizes['small'] = sizes[0]
        self.sizes['middle'] = sizes[1]
        self.sizes['large'] = sizes[2]
        self.totensor = ToTensor()
        self.random_crop = RandomCrop

    def __call__(self, data):
        nodule = {}
        for key, value in self.sizes.items():
            sample_size = [value[1], value[0], value[0]]
            crop = self.random_crop(sample_size)
            temp = crop(data)
            nodule[key] = self.totensor(temp)
        return nodule

class RandomCrop:
    def __init__(self, size):
        self.size = size
        self.randseed = np.floor(np.asarray(size) // 8)

    def __call__(self, data):
        s, y, x = data.shape

        des_s, des_y, des_x = self.size
        i = random.randint(-self.randseed[2], self.randseed[2])
        j = random.randint(-self.randseed[1], self.randseed[1])
        k = random.randint(-self.randseed[0], self.randseed[0])

        x_start = int(round((x - des_x) / 2.) + i)
        y_start = int(round((y - des_y) / 2.) + j)
        s_start = int(round((s - des_s) / 2.) + k)
        data = data[s_start: s_start + des_s,
               y_start: y_start + des_y,
               x_start: x_start + des_x]
        data = data[np.newaxis, :]
        return Tensor(data, mindspore.float32)

class RandomFlip:
    def __call__(self, data):
        if len(data.shape) == 3:
            base = 0
        elif len(data.shape) == 4:
            base = 1
        else:
            raise Exception('Random Flip Error!')
        if random.random() < 0.5:
            data = np.flip(data, base)
        if random.random() < 0.5:
            data = np.flip(data, base + 1)
        if random.random() < 0.5:
            data = np.flip(data, base + 2)

        return Tensor(data, mindspore.float32)

class RandomRotation:
    def __call__(self, data):
        assert len(data.shape) == 3, 'data shape: ' + str(data.shape)
        axial_rot_num = random.randint(0, 3)
        sag_rot_num = random.randint(0, 1)
        cor_rot_num = random.randint(0, 1)
        data = np.rot90(data, k=axial_rot_num, axes=(1, 2))
        data = np.rot90(data, k=sag_rot_num * 2, axes=(0, 1))
        data = np.rot90(data, k=sag_rot_num * 2, axes=(0, 2))
        return Tensor(data, mindspore.float32)
