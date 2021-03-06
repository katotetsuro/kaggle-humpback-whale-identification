from os.path import join
import torch.utils.data as data
from PIL import Image
import torchvision
import pandas as pd


class WhaleLabeledImageDataset(data.Dataset):
    """csvファイルからデータセットを作る。
    train_with_id.csvの作り方はnotebooks/split_dataset.ipynbを参照
    label=0のwhaleのバランスを調整する機能付き
    """

    def __init__(self, data_dir, transform=None):
        df = pd.read_csv(join(data_dir, 'train_with_id.csv'))
        self.image_dir = join(data_dir, 'train')

        # label=0のwhaleが多すぎるので、減らして見ようかな。。
        self.df_without_new_whale = df[df['label'] > 0]
        self.df_new_whale = df[df['label'] == 0]

        self.new_whale_size = 100
        self.new_whale_index = 0
        self.rotate_new_whale()
        if transform:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor

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

    def rotate_new_whale(self):
        last = self.new_whale_index + self.new_whale_size
        if last < len(self.df_new_whale):
            df = pd.concat(
                [self.df_without_new_whale, self.df_new_whale[self.new_whale_index:last]])
        else:
            df = pd.concat(
                [self.df_without_new_whale, self.df_new_whale[self.new_whale_index:]])
            last = last - len(self.df_new_whale)
            df = pd.concat([df, self.df_new_whale[:last]])

        self.new_whale_index += self.new_whale_size
        self.new_whale_index %= len(self.df_new_whale)
        self.df = df.reset_index(drop=True)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str


class CsvLabeledImageDataset(data.Dataset):
    """csvファイルからデータセットを作る。
    train_with_id.csvの作り方はnotebooks/split_dataset.ipynbを参照
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = img_dir

        if transform:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            Tensor image
            int label
        """
        # todo このパースをもっと汎用的に作る？
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
