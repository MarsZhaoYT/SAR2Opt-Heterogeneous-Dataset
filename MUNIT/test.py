"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--input_folder', type=str, help="input test dataset folder")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

# 定义输入测试集的路径，并获取路径下所有图像名称（未排序）
input_image_list = os.listdir(opts.input_folder)
root = opts.input_folder + os.sep

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    count = 0
    for image_name in input_image_list:
        count += 1
        tmp_image_path = root + image_name

        # 原始代码每次只处理1张图像进行测试，修改后可以测试目录下所有图像
        image = Variable(transform(Image.open(tmp_image_path).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

        # Start testing
        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            if opts.style != '':
                _, style = style_encode(style_image)
            else:
                style = style_rand
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_folder, image_name.split('.')[0] + '_output{:03d}.jpg'.format(j))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output.jpg')
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass

        if not opts.output_only:
            # also save input images
            # 同样保存原始图像，存储名为image_name
            vutils.save_image(image.data, os.path.join(opts.output_folder, image_name), padding=0, normalize=True)

        print('Processing ', count, 'th image')

    print('Test completed!')

end_time = time.time()
cost_time = end_time - start_time
print('cost time: ', cost_time)