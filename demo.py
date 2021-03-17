""" Demo script modified
    Generates forward and backward flow for cityscapes sequence.
    T0 = 19
    T1 = 19+[1..10]
    Uses two GPUs, one for fwd and the other for bwd
"""
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

# def norm_flow(flo, img_size):


def viz(img, img2, flo_fwd, flo_bwd, idx):
    img = img[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo_fwd = flo_fwd[0].permute(1,2,0).cpu().numpy()
    flo_bwd = flo_bwd[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo_fwd = flow_viz.flow_to_image(flo_fwd)
    flo_bwd = flow_viz.flow_to_image(flo_bwd)
    img_flo = np.concatenate([img, img2, flo_fwd, flo_bwd], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(f"flow_img/{idx}.png", img_flo[:, :, [2,1,0]])
    print(f"flow_img/{idx}.png")


def save_flow(flo, filename):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flo * 64.0 + 32768.0
    img = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=np.uint16)
    img[:, :, :2] = flo

    if not os.path.exists(os.path.dirname(filename)):
        print(f"make dir: {os.path.dirname(filename)}")
        os.makedirs(os.path.dirname(filename))

    if os.path.exists(filename):
        print(f"Warning: file {filename} exists! Will overwrite.")

    if not cv2.imwrite(filename, img):
        raise Exception(f"Can't write to {filename}!")

    print(f"Written {filename}.")


def demo(args):
    model = torch.nn.DataParallel(RAFT(args), device_ids=[0, 1], output_device=1).cuda()
    model.load_state_dict(torch.load(args.model))

    # model = model.module
    # model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))
        with open('seqheads.txt') as fs:
            image_seqheads = fs.read().splitlines()

        fileidx = 0
        for img1 in tqdm(image_seqheads):
            city, n1, n2, suf = img1.split('/', maxsplit=1)[-1].split("_")
            n2 = int(n2)
            for idx in range(1, 11):
                img2 = f"leftImg8bit_sequence/{city}_{n1}_{n2+idx:06d}_{suf}"
                savepath = f"flow_sequence/{city}_{n1}_{n2+idx:06d}"

                print(f"{img1}\n{img2}")

                image1 = load_image(os.path.join(args.path, img1))
                image2 = load_image(os.path.join(args.path, img2))
                savepath = os.path.join(args.path, savepath)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                image1_b = torch.cat([image1, image2], 0)
                image2_b = torch.cat([image2, image1], 0)

                flow_low, flow_up = model(image1_b, image2_b, iters=20, test_mode=True)
                flow_up_fwd = flow_up[0].unsqueeze(0)
                flow_up_bwd = flow_up[1].unsqueeze(0)
                # viz(image1, image2, flow_up_fwd, flow_up_bwd, fileidx)
                save_flow(flow_up_fwd, f"{savepath}_flow_fwd.png")
                save_flow(flow_up_bwd, f"{savepath}_flow_bwd.png")

                fileidx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
