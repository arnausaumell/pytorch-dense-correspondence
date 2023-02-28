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
import torch.nn.functional as F
from sklearn.decomposition import PCA

import pytorch_dense_correspondence.modules.dense_correspondence_manipulation.utils.utils as utils
import pytorch_dense_correspondence.dense_correspondence.evaluation.plotting as dc_plotting
from pytorch_dense_correspondence.dense_correspondence.evaluation.evaluation import *


def set_up_model(network_name="shirt_hanging_d16_distributional_sym_rot"):
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_filename = os.path.join(
        dc_source_dir, "config/dense_correspondence/evaluation/evaluation.yaml"
    )
    config = utils.getDictFromYamlFilename(config_filename)
    dcn = DenseCorrespondenceEvaluation(config).load_network_from_config(network_name)
    dataset = dcn.load_training_dataset()
    return dataset, dcn


def get_canonical_image():
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    rgb_filename = os.path.join(
        dc_source_dir,
        "pdc/logs_proto/shirt_canonical/processed/images/rgb-cam1-0-0.png",
    )
    return np.array(Image.open(rgb_filename).convert("RGB"))


def correspondence_finder(
    rgb_a, rgb_b, pixel_a, mask_b=None, visualize=False, plot_save_dir=None
):
    dataset, dcn = set_up_model()

    # transform to cv2.rgb_image to np.array
    rgb_a_array = np.array(rgb_a)
    rgb_b_array = np.array(rgb_b)

    # compute dense descriptors
    rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
    rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

    # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
    res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu()
    res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu()

    pixel_b, _, norm_diffs = DenseCorrespondenceNetwork.find_best_match(
        pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b
    )

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    # Plot original images
    rgb_a = cv2.circle(rgb_a, pixel_a, 10, (255, 0, 0), -1)
    rgb_b = cv2.circle(rgb_b, pixel_b, 10, (255, 0, 0), -1)
    ax[0, 0].imshow(rgb_a)
    ax[0, 0].set_title("Img A")
    ax[0, 1].imshow(rgb_b)
    ax[0, 1].set_title("Img B")

    # PCA plots
    d = res_a.shape[-1]
    res_shape = res_a.shape
    pca = PCA(n_components=3)

    res_b_pc = pca.fit_transform(res_b.reshape((-1, d)))
    res_b_pc = res_b_pc.reshape(res_shape[0], res_shape[1], -1)
    res_b_pc = dc_plotting.normalize_descriptor(res_b_pc)
    ax[1, 1].imshow(res_b_pc)
    ax[1, 1].set_title("Img B: PCA plot DOD")

    res_a_pc = pca.transform(res_a.reshape((-1, d)))
    res_a_pc = res_a_pc.reshape(res_shape[0], res_shape[1], -1)
    res_a_pc = dc_plotting.normalize_descriptor(res_a_pc)
    ax[1, 0].imshow(res_a_pc)
    ax[1, 0].set_title("Img A: PCA plot DOD")

    # Heatmap plots
    norm_diffs = torch.tensor(norm_diffs)
    p_b = F.softmax(
        -1 * norm_diffs.ravel(), dim=0
    ).double()  # compute current distribution
    p_b = p_b.reshape(rgb_b_tensor.shape[1:]).numpy()
    confidence_level = p_b[pixel_b[1], pixel_b[0]]

    data_sorted = np.sort(p_b.ravel())
    p = 1.0 * np.arange(len(p_b.ravel())) / (len(p_b.ravel()) - 1)
    ax[1, 2].plot(p, data_sorted)
    ax[1, 2].set_ylabel("Probability across pixels")
    ax[1, 2].set_xlabel("Percentage of image pixels")

    im1 = ax[0, 2].imshow(p_b, cmap="jet", alpha=0.95)  # , vmin=0, vmax=0.005)
    im2 = ax[0, 2].imshow(rgb_b_array, alpha=0.15)
    plt.colorbar(im1, ax=ax[0, 2], fraction=0.046, pad=0.04)
    ax[0, 2].set_title("Img B: Best match heatmap")

    plt.tight_layout()
    if plot_save_dir:
        plt.savefig(os.path.join(plot_save_dir, "analysis_plots.png"))
    if visualize:
        plt.show()

    return pixel_b, confidence_level


def fill_gaps_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def masking(camera_color_img):
    input_filename = "/home/gelsight/Desktop/camera_img.png"
    output_filename = "/home/gelsight/Desktop/masked_img.png"
    cv2.imwrite(input_filename, camera_color_img)
    os.system("backgroundremover -i %s -o %s" % (input_filename, output_filename))
    camera_color_img_masked = cv2.imread(output_filename)

    mask = (camera_color_img_masked != [0, 0, 0])[:, :, 0].astype("uint8")
    mask = fill_gaps_mask(mask)
    return camera_color_img_masked, mask


def correct_mask_with_depth(original_mask, camera_depth_img):
    # Delete from the original_mask those points with "infinite" depth
    mask_depth = (camera_depth_img < 930).astype("uint8")
    mask_depth *= (camera_depth_img > 500).astype("uint8")

    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(mask_depth)
    # ax[0, 1].imshow(original_mask)
    # ax[1, 0].imshow(original_mask * mask_depth)
    # ax[1, 0].set_title("Final mask")
    # plt.show()
    return original_mask * mask_depth
