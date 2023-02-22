import cv2
import numpy as np
import os
import argparse
import scipy
from PIL import Image


def preprocess_dataset(input_dir, out_filenames, top=0, left=100, right=1380, bottom=960, output_size=(960, 720)):
    for filename in os.listdir(input_dir):
        idx = filename.split('_')[-1].split('.')[0]
        filename = os.path.join(input_dir, filename)
        if 'png' in filename:
            if 'color' in filename:
                img = Image.open(filename)
                img = img.crop((left, top, right, bottom))
                img = img.resize(output_size)
                img.save(out_filenames['rgb_img'] % idx)
            if 'depth' in filename:
                img = Image.open(filename)
                img = img.crop((left, top, right, bottom))
                img = img.resize(output_size)
                img.save(out_filenames['depth_img'] % idx)
        
        elif 'npy' in filename:
            depth_val = np.load(filename)
            np.save(out_filenames['depth_val'] % idx, depth_val)
            mask = (depth_val < 930).astype(int)
            mask *= (depth_val > 100).astype(int)
            mask[:, :500] = 0
            mask[:, 1200:] = 0
            mask[900:, :] = 0
            mask = mask[top:bottom, left:right]
            mask_visible = mask*255
            mask = Image.fromarray(mask.astype(np.uint8)).resize(output_size)
            mask_visible = Image.fromarray(mask_visible.astype(np.uint8)).resize(output_size)

            mask.save(out_filenames['mask'] % idx)
            mask_visible.save(out_filenames['visible_mask'] % idx)
    
    for filename in os.listdir(input_dir):
        idx = filename.split('_')[-1].split('.')[0]
        filename = os.path.join(input_dir, filename)
        if 'png' in filename:
            if 'color' in filename:
                img = cv2.imread(out_filenames['rgb_img'] % idx, 1)
                mask = cv2.imread(out_filenames['mask'] % idx, 0)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                img_masked = cv2.bitwise_and(img, img, mask=mask)               
                cv2.imwrite(out_filenames['rgb_img'] % idx, img_masked)


if __name__ == '__main__':
    
    # argParser = argparse.ArgumentParser()
    # argParser.add_argument("-s", "--scene_name", help="name of the scene")
    # args = argParser.parse_args()
    
    for input_dir in os.listdir('./'):
        if '0120' in input_dir and not 'real' in input_dir:
            print('Processing %s...' % input_dir)
            output_dir = 'real_data_%s/processed' % input_dir
            images_dir    = os.path.join(os.getcwd(), output_dir, 'images')
            depth_val_dir = os.path.join(os.getcwd(), output_dir, 'depth_values')
            masks_dir     = os.path.join(os.getcwd(), output_dir, 'image_masks')

            # Masking using depth images
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(depth_val_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            filenames = {'rgb_img':     os.path.join(images_dir, 'rgb-%s.png'),
                        'depth_img':    os.path.join(images_dir, 'depth-%s.png'),
                        'depth_val':    os.path.join(depth_val_dir, 'depth-%s.npy'),
                        'mask':         os.path.join(masks_dir, 'mask-%s.png'),
                        'visible_mask': os.path.join(masks_dir, 'visible_mask-%s.png')}

            preprocess_dataset(input_dir, filenames)

