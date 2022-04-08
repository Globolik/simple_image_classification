import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np

class VegitablesDataset(Dataset):
    def __init__(self, csv_annotations, root_dir, transform, train):
        self.annotations = pd.read_csv(csv_annotations)
        self.root_dir = root_dir+'/'+'train' if train else root_dir+'/'+'test'
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        image = np.array(image)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)

        return (image, label, image_path)



if __name__ == '__main__':
    from torchvision.utils import save_image
    import shutil

    mean = (119, 117, 87)
    std = (51, 51, 49)
    mean = tuple(round(_ / 255, 5) for _ in mean)
    std = tuple(round(_ / 255, 5) for _ in std)

    transforms_to_do = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomApply(
                [
                    transforms.RandomCrop((190, 190)),
                    transforms.RandomRotation(degrees=30,),
                    transforms.ColorJitter(brightness=0.1),

                ],
                p=0.5),
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])



    dataset = VegitablesDataset(
        csv_annotations='annotations.csv',
        root_dir='/home/globolik/simple_torch_nn/vegetables/Vegetable Images',
        transform=transforms_to_do,
        train=True

    )

    for _ in range(112, 120):
        print('_', _)
        image, label, orig_image  = dataset.__getitem__(_)
        save_image(image, f'test_images_augm/current_test/im{_}.jpeg')
        shutil.copyfile(orig_image, f'test_images_augm/current_test/orig{_}.jpeg')


