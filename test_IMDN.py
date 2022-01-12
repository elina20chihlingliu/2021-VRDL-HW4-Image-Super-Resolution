import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture
# Testing settings

parser = argparse.ArgumentParser(description='IMDN')

parser.add_argument("--test_lr_folder", type=str, default = 'dataset/testing_lr_images/testing_lr_images',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default = 'results/')
parser.add_argument("--checkpoint", type=str, default = 'epoch_100.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default = True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default = 3,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')


model = architecture.IMDN(upscale=opt.upscale_factor)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
filelist = os.listdir(opt.test_lr_folder)
for imname in filelist:
    im_l = cv2.imread(os.path.join(opt.test_lr_folder, imname), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    if opt.is_y is True:
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
   

    output_folder = os.path.join(opt.output_folder,
                                 imname[:-4] + '_pred.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1

