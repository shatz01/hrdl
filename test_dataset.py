import os
import torch
import torchvision
from src.data_stuff.NEW_patch_dataset import PatchDataset

# --- hypers --- #
data_dir = '/home/shatz/repos/data/imagenette2_tesselated'
batch_size = 4
group_size = 3
num_workers = 4
# ------------- #

train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'val')

# transforms
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
train_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])

# dataset
train_ds = PatchDataset(train_data_dir, group_size=group_size, transform=train_transforms)
val_ds = PatchDataset(val_data_dir, group_size=group_size, transform=train_transforms)
print("--- Dataset sample:", end="")
print(train_ds[0])

import pdb; pdb.set_trace()



# dataloaders
train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
        )
val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True
        )

print("--- DataLoader sample:", end="")
dl_iter = next(iter(train_dl))
