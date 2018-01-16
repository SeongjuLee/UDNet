from UDNet import *
import numpy as np
import tifffile as tiff
from PIL import Image
import glob

batch_size = 1
img_size = 512
img_dir = "./test_img"

try:
    fusion = torch.load('./model/model.pkl')
    print("[INFO] Model Restored")
except:
    print("[INFO] Model Not Restored")
    pass

for i in range(len(glob.glob('./test_img/biomedical/*.jpg'))):
    im = Image.open('./test_img/biomedical/{}.jpg'.format(i))
    im = np.asarray(im)
    im = im[:, :, 2]
    im = np.reshape(im, (1, 1, 512, 512))/255.0
    im = torch.from_numpy(im).float()

    x = Variable(im).cuda()
    y = fusion(x)
    y = torch.round(y).int()

    x_img = x[0].cpu().data
    y_img = y[0].cpu().data
    # print(np.shape(y_img.numpy()))
    if i == 0:
        result = y.cpu().data.numpy()
        print(np.shape(result))
    else:
        print(np.shape(y.cpu().data.numpy()))
        result = np.concatenate((result, y.cpu().data.numpy()), axis=0)

    v_utils.save_image(x_img, "./test_result/{}_input.png".format(i))
    v_utils.save_image(y_img, "./test_result/{}_output.png".format(i))

print(np.shape(result))
result = result * 255
tiff.imsave('./submit.tif', result.astype(np.uint8))

# img = tiff.imread('./ISBI/test-volume.tif')
# print(np.shape(img))
#
# cv.imwrite('x.tif', result*255)
# for i in range(30):
#     img_rgb = cv.cvtColor(img[i], cv.COLOR_GRAY2RGB)
#     cv.imwrite('./test_img/{}.jpg'.format(i), img_rgb)

