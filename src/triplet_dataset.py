from os.path import join
import torch.utils.data as data
from PIL import Image
import torchvision
import pandas as pd
from random import random
import numpy as np


class TripletDataset(data.Dataset):
    """
    train_with_id.csvの作り方はnotebooks/split_dataset.ipynbを参照
    """

    def __init__(self, data_dir, transform=None, new_whale_prob=0.1):
        df = pd.read_csv(join(data_dir, 'train_with_id.csv'))
        self.image_dir = join(data_dir, 'train')

        # 2枚以上の画像がある個体に絞る
        self.df_without_new_whale = df[df.label > 0]
        _, counts = np.unique(
            df.label, return_counts=True)
        self.df_without_new_whale = self.df_without_new_whale[
            counts[self.df_without_new_whale.label] > 1]
        self.df_new_whale = df[df.label == 0]

        self.prob = new_whale_prob

        if transform:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            Tensor image
            int label
        """
        img_file, _, label = self.df_without_new_whale.iloc[index]
        anchor = Image.open(join(self.image_dir, img_file)).convert('RGB')

        img_file, _, pos_label = self.df_without_new_whale[self.df_without_new_whale.label == label].sample(
        ).iloc[0]
        positive = Image.open(
            join(self.image_dir, img_file)).convert('RGB')

        if random() < self.prob:
            # new_whaleから選択
            img_file, _, neg_label = self.df_new_whale.sample().iloc[0]
        else:
            # labelが異なるデータから選択
            img_file, _, neg_label = self.df_without_new_whale[self.df_without_new_whale.label != label].sample(
            ).iloc[0]

        negative = Image.open(
            join(self.image_dir, img_file)).convert('RGB')

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.df_without_new_whale)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str
