import argparse
import numpy as np
from maps.scripts.utils import split_image
from tqdm import tqdm
from ml_logger import logger
import cv2
from PIL import Image
import os
from utils.basic import np_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--cap_size', type=int, default=100)
    parser.add_argument('--out_path', default='maps.npz')
    parser.add_argument('--new_size', type=int, default=128)
    args = parser.parse_args()
    res = 512
    step = 600

    maps = []
    sats = []
    it = 0

    for filename in tqdm(os.listdir(args.path)):
        if not filename.endswith('.png'):
            continue

        img = Image.open(os.path.join(args.path, filename))
        arr = np.array(img, dtype=np.uint8)
        assert(arr.shape == (step, 2 * step, 3))
        sat = arr[0:res, 0:res]
        map = arr[0:res, step:step+res]

        def add(lst, img):
            lst.append(split_image(cv2.resize(img, (args.new_size, args.new_size), interpolation=cv2.INTER_AREA), size=args.size, stride=args.stride))

        add(maps, map)
        add(sats, sat)
        it += 1
        if it >= args.cap_size:
            break

    save_dict = {
        'map': np.concatenate(maps),
        'sat': np.concatenate(sats)}

    np_stats(save_dict['map'], 'map')
    np_stats(save_dict['sat'], 'sat')
    np.savez(args.out_path, **save_dict)


if __name__ == '__main__':
    main()
