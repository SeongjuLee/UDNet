import torch
from UDNet import *
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# hyperparameters
batch_size = 1
img_size = 512

# input pipeline

img_dir = "./merged/"
# img_data = dset.ImageFolder(root=img_dir, transform=transforms.Compose([
#     transforms.Resize(size=img_size),
#     transforms.CenterCrop(size=(img_size, img_size * 2)),
#     transforms.ToTensor(),
# ]))
#
# img_batch = data.DataLoader(img_data, batch_size=batch_size,
#                             shuffle=True, num_workers=2)

val_data = dset.ImageFolder(root=img_dir, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
]))

val_batch = data.DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=2)

# initiate FusionNet
try:
    fusion = torch.load('./model/model.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

dice_list = []
for _, (image, label) in enumerate(val_batch):
    satel_image, map_image = torch.chunk(image, chunks=2, dim=3)

    x = Variable(satel_image).cuda()
    y_ = Variable(map_image).cuda()
    y = fusion(x)
    print(x)
    v_utils.save_image(x[0].cpu().data, "./test/{}_input.png".format(_))
    v_utils.save_image(y_[0].cpu().data, "./test/{}_label.png".format(_))
    v_utils.save_image(y[0].cpu().data, "./test/{}_output.png".format(_))

    y = torch.mean(y, 1)
    y_ = torch.mean(y_, 1)

    y = torch.round(y).int()
    y_ = torch.round(y_).int()

    mul = y * y_

    for k in range(batch_size):
        overlap = torch.nonzero(mul[k]).size(0)
        y_true = torch.nonzero(y_[k]).size(0)
        y_pred = torch.nonzero(y[k]).size(0)

        dice = 2.0 * overlap / (y_true + y_pred)
        print(dice)
        dice_list.append(dice)

print(len(dice_list))
print('Dice: {}'.format(np.average(dice)))
