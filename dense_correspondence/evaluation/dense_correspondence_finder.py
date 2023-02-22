import os
import os, sys

# sys.path.append(os.getcwd() + "/../../modules")
# sys.path.append(os.getcwd() + "/../../external")
# sys.path.append(os.getcwd() + "/../..")
# sys.path.append(os.getcwd() + "/../dataset")
sys.path.append(os.getcwd() + "/pytorch_dense_correspondence")
sys.path.append(os.getcwd() + "/pytorch_dense_correspondence/modules")
sys.path.append(os.getcwd() + "/pytorch_dense_correspondence/external")
sys.path.append(
    os.getcwd() + "/pytorch_dense_correspondence/dense_correspondence/dataset"
)

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import torch
import pytorch_dense_correspondence.modules.dense_correspondence_manipulation.utils.utils as utils

from dense_correspondence.evaluation.evaluation import *
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType

dc_source_dir = utils.getDenseCorrespondenceSourceDir()
config_filename = os.path.join(
    dc_source_dir, "config/dense_correspondence/evaluation/evaluation.yaml"
)
config = utils.getDictFromYamlFilename(config_filename)

dce = DenseCorrespondenceEvaluation(config)

network_name = "shirt_hanging_d16_distributional_sym"
dcn = dce.load_network_from_config(network_name)
dataset = dcn.load_training_dataset()


def get_canonical_image():
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    rgb_filename = os.path.join(
        dc_source_dir,
        "pdc/logs_proto/shirt_canonical/processed/images/rgb-cam1-0-0.png",
    )
    return np.array(Image.open(rgb_filename).convert("RGB"))


def correspondence_finder(rgb_a, rgb_b, pixel_a):

    rgb_a_array = np.array(rgb_a)
    rgb_b_array = np.array(rgb_b)

    # mask_a = np.asarray(mask_a)
    # mask_b = np.asarray(mask_b)

    # compute dense descriptors
    rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
    rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

    # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
    res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu()
    res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu()

    pixel_b, _, norm_diffs = DenseCorrespondenceNetwork.find_best_match(
        pixel_a, res_a.numpy(), res_b.numpy(),
    )
    # uv_b_masked, _, _ = DenseCorrespondenceNetwork.find_best_match(
    #     pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b
    # )
    return pixel_b


if __name__ == "__main__":
    rgb_img_a = get_canonical_image()
    rgb_img_b = get_canonical_image()
    print(correspondence_finder(rgb_img_a, rgb_img_b, (320, 210)))
