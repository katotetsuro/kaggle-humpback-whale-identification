from torchvision import transforms

"""
jupyter notebookとtrainスクリプトの両方でで同じtransformにしたいので、
どっちもこれを読み込んで使うようにしようかな・・？
"""


def get_transform():
    data_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10,  scale=(0.8, 1.2),
                                shear=0.2, resample=Image.BILINEAR),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    return data_transform
