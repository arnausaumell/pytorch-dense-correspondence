import cv2
import numpy as np
import os
import argparse
import scipy
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def create_masks(masked_images_dir, out_filenames):
    for filename in os.listdir(masked_images_dir):
        masked_img = cv2.imread(os.path.join(masked_images_dir, filename))
        mask = (masked_img != [0, 0, 0])[:,:,0].astype('uint8')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        idx = filename.split('-')[-1].split('.')[0]
        cv2.imwrite(out_filenames['mask'] % idx, mask)
        cv2.imwrite(out_filenames['visible_mask'] % idx, mask*255)        


def preprocess_dataset(real_images_dir, masked_images_dir, out_filenames, top=0, left=100, right=1380, bottom=960, output_size=(960, 720)):
    for filename in os.listdir(masked_images_dir):
        idx = filename.split('-')[-1].split('.')[0]
        filename = os.path.join(masked_images_dir, filename)
        if 'png' in filename:
            if 'rgb' in filename:
                img = Image.open(filename)
                img = img.crop((left, top, right, bottom))
                img = img.resize(output_size)               

                mask = Image.open(out_filenames['mask'] % idx)
                mask = mask.crop((left, top, right, bottom))
                mask = mask.resize(output_size)
                mask.save(out_filenames['mask'] % idx)

                mask = np.repeat(np.array(mask)[:, :, np.newaxis], 3, axis=2)
                img = np.array(img)[:,:,:3]
                img = Image.fromarray(img*mask)
                img.save(out_filenames['rgb_img'] % idx)     

                visible_mask = Image.open(out_filenames['visible_mask'] % idx)
                visible_mask = visible_mask.crop((left, top, right, bottom))
                visible_mask = visible_mask.resize(output_size)
                visible_mask.save(out_filenames['visible_mask'] % idx)
                
                # masked_image = Image.composite(255, img, ImageOps.invert(visible_mask))
                # masked_image.save(out_filenames['rgb_img'] % idx)
        
    for filename in real_images_dir: 
        idx = filename.split('_')[-1].split('.')[0]
        if 'depth' in filename:
            img = Image.open(filename)
            img = img.crop((left, top, right, bottom))
            img = img.resize(output_size)
            img.save(out_filenames['depth_img'] % idx)
    
    


if __name__ == '__main__':    
    for shirt_name in os.listdir('./masked_images'):
        print('Processing %s...' % shirt_name)
        
        real_images_dir   = 'real_images/%s' % shirt_name
        masked_images_dir = 'masked_images/%s' % shirt_name
        
        output_dir = '%s/processed' % shirt_name
        images_dir    = os.path.join(os.getcwd(), output_dir, 'images')
        depth_val_dir = os.path.join(os.getcwd(), output_dir, 'depth_values')
        masks_dir     = os.path.join(os.getcwd(), output_dir, 'image_masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(depth_val_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        filenames = {'rgb_img':     os.path.join(images_dir, 'rgb-%s.png'),
                     'depth_img':    os.path.join(images_dir, 'depth-%s.png'),
                     'depth_val':    os.path.join(depth_val_dir, 'depth-%s.npy'),
                     'mask':         os.path.join(masks_dir, 'mask-%s.png'),
                     'visible_mask': os.path.join(masks_dir, 'visible_mask-%s.png')}

        # Create masks from masked images
        create_masks(masked_images_dir, filenames)
        # Create rest of the dataset
        preprocess_dataset(real_images_dir, masked_images_dir, filenames)

