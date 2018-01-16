import torch
from UDNet import *
import numpy as np
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 1
batch_size_val = 1
img_size = 512
lr = 0.1
epoch = 1000


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, true):
        return 1. - DiceScore(pred, true)


def DiceScore(pred, true):
    pred = torch.round(pred).int()
    true = torch.round(true).int()
    matmul = pred * true

    pred = torch.sum(pred, dim=3)
    pred = torch.sum(pred, dim=2).float()

    true = torch.sum(true, dim=3)
    true = torch.sum(true, dim=2).float()

    matmul = torch.sum(matmul, dim=3)
    matmul = torch.sum(matmul, dim=2).float()

    dice_tensor = 2. * matmul / (pred + true)
    dice_score = torch.mean(dice_tensor)

    return dice_score


def tobinary(img):
    img = torch.round(img).int()
    return img


# input pipeline
img_dir = "./merged/"
img_data = dset.ImageFolder(root=img_dir, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
]))

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

val_dir = "./original/"
val_data = dset.ImageFolder(root=val_dir, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
]))

val_batch = data.DataLoader(val_data, batch_size=batch_size_val,
                            shuffle=False, num_workers=2)

# initiate UDNet
UDNet = nn.DataParallel(UDNetGenerator(1, 1, 64)).cuda()

try:
    fusion = torch.load('./model/fusion.pkl')
    print("[INFO] Model Restored")
except:
    print("[INFO] Model Not Restored")
    pass

# loss function & optimizer
loss_func1 = nn.SmoothL1Loss()
loss_func2 = DiceLoss()

optimizer = torch.optim.SGD(UDNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)

# loss and dice score data file open
f_loss = open('loss.txt', 'w')
f_dice = open('dice.txt', 'w')

# training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
for i in range(epoch):
    scheduler.step()
    losslist = []
    for _, (image, label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3)
        optimizer.zero_grad()

        x = Variable(satel_image).cuda()
        y_ = Variable(map_image).cuda()
        y = UDNet.forward(x)

        loss = 10.*loss_func1(y, y_) + loss_func2(y, y_)
        loss.backward()
        optimizer.step()
        losslist.append(loss.cpu().data[0])

        print('Epoch: {}, Batch: {}/{}, MiniBatchLoss: {}'.format(i, (_ + 1), (len(img_batch)), loss.cpu().data[0]))

    loss_average = np.average(losslist)
    f_loss.write(str(loss_average))
    f_loss.write('\n')
    print('Epoch: {}, Loss: {}'.format(i, loss_average))

    dicelist = []
    for _, (image, label) in enumerate(val_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3)

        x = Variable(satel_image).cuda()
        true = Variable(map_image).cuda()
        pred = UDNet(x)

        true_binary = tobinary(true)
        pred_binary = tobinary(pred)

        dice = DiceScore(true, pred).cpu().data
        dicelist.append(dice)

        if _ == 0 and i == 0:
            v_utils.save_image(true[0].cpu().data, "./result/{}_label_epoch{}.png".format(_, i))
        if _ == 0:
            v_utils.save_image(pred[0].cpu().data, "./result/{}_output_epoch{}.png".format(_, i))
            v_utils.save_image(pred_binary[0].cpu().data, "./result/{}_output_binary_epoch{}.png".format(_, i))

    dice_average = np.mean(dicelist)
    f_dice.write(str(dice_average))
    f_dice.write('\n')
    print('Dice: {}'.format(dice_average))
    if i % 50 == 0:
        torch.save(UDNet, "./model/model{}_{}.pkl".format(i, dice_average))

# file close
f_loss.close()
f_dice.close()
