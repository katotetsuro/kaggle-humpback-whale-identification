from os.path import join
import torch.utils.data as data
from PIL import Image
import torchvision
import pandas as pd
from random import random
import numpy as np
import random

"""
https://arxiv.org/pdf/1703.07737.pdf
同じクラスからimage_per_class枚ずつサンプリングするバッチを作るためのデータセット
batchsize/image_per_class クラスのサンプルが1バッチ内に渡せる
"""


class OnlineMiningDataset(data.Dataset):
    """
    train_with_id.csvの作り方はnotebooks/split_dataset.ipynbを参照
    """

    def __init__(self, data_dir, transform=None, image_per_class=4, limit=None, min_size=1, exclude_new_whale=True, batch_size=128):
        df = pd.read_csv(join(data_dir, 'train_with_id.csv'))
        self.image_dir = join(data_dir, 'cropped/train')

        # min_size枚以上の画像がある個体に絞る
        # todo df_without_new_whaleという変数名は、exclude_new_whale=Falseのときに意味不明になるな
        self.df_without_new_whale = df[df.label >
                                       0] if exclude_new_whale else df
        _, self.counts = np.unique(
            df.label, return_counts=True)
        self.df_without_new_whale = self.df_without_new_whale[
            self.counts[self.df_without_new_whale.label] >= min_size]
        self.df_without_new_whale = self.df_without_new_whale.reset_index(
            drop=True)
        print('最低{}枚はサンプルがあるクラスだけを使う。対象サンプル数:{}'.format(
            min_size, len(self.df_without_new_whale)))
        self.df_new_whale = df[df.label == 0]

        if transform:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor()

        self.image_per_class = image_per_class
        self.batch_size = batch_size

        self.sample()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            Tensor image
            int label
        """
        img_file, _, label = self.df.iloc[index]
        img = Image.open(join(self.image_dir, img_file)).convert('RGB')
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str

    def sample(self):
        dfs = []
        labels = np.unique(self.df_without_new_whale.label)
        random.shuffle(labels)

        for _ in range(max(1, 2000//len(labels))):
            for l in labels:
                d = self.df_without_new_whale[self.df_without_new_whale.label == l]
                replace = len(d) < self.image_per_class
                dfs.append(d.sample(self.image_per_class, replace=replace))
                if ((len(dfs) + 1) * self.image_per_class) % self.batch_size == 0:
                    dfs.append(self.df_new_whale.sample(
                        self.image_per_class, replace=False))

        df = pd.concat(dfs)
        self.df = df.reset_index(drop=True)
