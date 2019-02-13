from torchvision import transforms
from PIL import Image


"""
jupyter notebookとtrainスクリプトの両方でで同じtransformにしたいので、
どっちもこれを読み込んで使うようにしようかな・・？
"""


def get_train_transform():
    data_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.3),
        transforms.ColorJitter(
            brightness=20, contrast=20, saturation=20, hue=-0.5),
        transforms.RandomAffine(10,  scale=(0.8, 1.2),
                                shear=0.2, resample=Image.BILINEAR),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return data_transform


def get_test_transform():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return data_transform
