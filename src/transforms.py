from torchvision import transforms
from PIL import Image


"""
jupyter notebookとtrainスクリプトの両方でで同じtransformにしたいので、
どっちもこれを読み込んで使うようにしようかな・・？
"""


def get_transform():
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return data_transform
