#!/usr/bin/python
import logging
import os
import random

import dense_correspondence_manipulation.utils.utils as utils

utils.add_dense_correspondence_to_python_path()
import matplotlib.pyplot as plt
import cv2
import itertools
import numpy as np
import pandas as pd
import random
import scipy.stats as ss
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

import plotly
import plotly.express as px
import plotly.graph_objects as xgo
import pandas as pd
import seaborn as sns

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from dense_correspondence_manipulation.utils.constants import *
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
from dense_correspondence.loss_functions.pixelwise_distributional_loss import PixelwiseDistributionalLoss

import dense_correspondence.evaluation.plotting as dc_plotting

from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image

COLOR_BLUE = (0, 0, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 100, 0)


class PandaDataFrameWrapper(object):
    """
    A simple wrapper for a PandaSeries that protects from read/write errors
    """

    def __init__(self, columns):
        data = [np.nan] * len(columns)
        self._columns = columns
        self._df = pd.DataFrame(data=[data], columns=columns)

    def set_value(self, key, value):
        if key not in self._columns:
            raise KeyError("%s is not in the index" % (key))

        self._df[key] = value

    def get_value(self, key):
        return self._df[key]

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, value):
        self._series = value


class DCNEvaluationPandaTemplate(PandaDataFrameWrapper):
    columns = ['scene_name',
               'img_a_idx',
               'img_b_idx',
               'is_valid',
               'is_valid_masked',
               'norm_diff_descriptor_ground_truth',
               'norm_diff_descriptor',
               'norm_diff_descriptor_masked',
               'norm_diff_ground_truth_3d',
               'norm_diff_pred_3d',
               'norm_diff_pred_3d_masked',
               'pixel_match_error_l2',
               'pixel_match_error_l2_masked',
               'pixel_match_error_l1',
               'fraction_pixels_closer_than_ground_truth',
               'fraction_pixels_closer_than_ground_truth_masked',
               'average_l2_distance_for_false_positives',
               'average_l2_distance_for_false_positives_masked']

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNEvaluationPandaTemplate.columns)


class DCNEvaluationPandaTemplateAcrossObject(PandaDataFrameWrapper):
    columns = ['scene_name_a',
               'scene_name_b',
               'img_a_idx',
               'img_b_idx',
               'object_id_a',
               'object_id_b',
               'norm_diff_descriptor_best_match']

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, DCNEvaluationPandaTemplateAcrossObject.columns)


# +

class DenseCorrespondenceEvaluation(object):
    """
    Samples image pairs from the given scenes. Then uses the network to compute dense
    descriptors. Records the results of this in a Pandas.DataFrame object.
    """

    def __init__(self, config):
        self._config = config
        self._dataset = None

    def load_network_from_config(self, name):
        """
        Loads a network from config file. Puts it in eval mode by default
        :param name:
        :type name:
        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" % (name))

        path_to_network_params = self._config["networks"][name]["path_to_network_params"]
        path_to_network_params = utils.convert_to_absolute_path(path_to_network_params)
        model_folder = os.path.dirname(path_to_network_params)

        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        dcn.eval()
        return dcn

    def load_dataset_for_network(self, network_name):
        """
        Loads a dataset for the network specified in the config file
        :param network_name: string
        :type network_name:
        :return: SpartanDataset
        :rtype:
        """
        if network_name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" % (network_name))

        network_folder = os.path.dirname(self._config["networks"][network_name]["path_to_network_params"])
        network_folder = utils.convert_to_absolute_path(network_folder)
        dataset_config = utils.getDictFromYamlFilename(os.path.join(network_folder, "dataset.yaml"))

        dataset = SpartanDataset(config=dataset_config)
        return dataset

    def load_dataset(self):
        """
        Loads a SpartanDatasetMasked object
        For now we use a default one
        :return:
        :rtype: SpartanDatasetMasked
        """

        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'dataset',
                                   'spartan_dataset_masked.yaml')

        config = utils.getDictFromYamlFilename(config_file)

        dataset = SpartanDataset(mode="test", config=config)

        return dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def get_output_dir(self):
        return utils.convert_to_absolute_path(self._config['output_dir'])

    @staticmethod
    def get_image_pair_random(dataset, scene_name):
        img_a_idx = dataset.get_random_image_index(scene_name)
        img_b_idx = dataset.get_random_image_index('shirt_canonical')
        return (img_a_idx, img_b_idx)

    @staticmethod
    def evaluate_network(dcn, dataset, num_image_pairs=25, num_matches_per_image_pair=100):
        """

        :param nn: A neural network DenseCorrespondenceNetwork
        :param test_dataset: DenseCorrespondenceDataset
            the dataset to draw samples from
        :return:
        """
        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation
        dcn.eval()

        logging_rate = 5

        pd_dataframe_list = []
        for i in range(0, num_image_pairs):

            scene_name = dataset.get_random_scene_name()

            # grab random scene
            if i % logging_rate == 0:
                print("computing statistics for image %d of %d, scene_name %s" % (i, num_image_pairs, scene_name))
                print("scene")

            idx_pair = DCE.get_image_pair_random(dataset, scene_name)

            if idx_pair is None:
                logging.info("no satisfactory image pair found, continuing")
                continue

            img_idx_a, img_idx_b = idx_pair

            dataframe_list_temp = \
                DCE.single_same_scene_image_pair_quantitative_analysis(dcn, dataset, scene_name,
                                                                       img_idx_a,
                                                                       img_idx_b,
                                                                       num_matches=num_matches_per_image_pair,
                                                                       debug=False)

            if dataframe_list_temp is None:
                print("no matches found, skipping")
                continue

            pd_dataframe_list += dataframe_list_temp
            # pd_series_list.append(series_list_temp)

        df = pd.concat(pd_dataframe_list)
        return pd_dataframe_list, df

    @staticmethod
    def plot_descriptor_colormaps(res_a, res_b, descriptor_image_stats=None,
                                  mask_a=None, mask_b=None, plot_masked=False, descriptor_norm_type="mask_image"):
        """
        Plots the colormaps of descriptors for a pair of images
        :param res_a: descriptors for img_a
        :type res_a: numpy.ndarray
        :param res_b:
        :type res_b: numpy.ndarraya
        :param descriptor_norm_type: what type of normalization to use for the
        full descriptor image
        :type : str
        :return: None
        :rtype: None
        """

        if plot_masked:
            nrows = 2
            ncols = 2
        else:
            nrows = 1
            ncols = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(5)
        fig.set_figwidth(15)

        if descriptor_image_stats is None:
            res_a_norm, res_b_norm = dc_plotting.normalize_descriptor_pair(res_a, res_b)
        else:
            res_a_norm = dc_plotting.normalize_descriptor(res_a, descriptor_image_stats[descriptor_norm_type])
            res_b_norm = dc_plotting.normalize_descriptor(res_b, descriptor_image_stats[descriptor_norm_type])

        if plot_masked:
            ax = axes[0, 0]
        else:
            ax = axes[0]
        
        ax.imshow(res_a_norm)

        if plot_masked:
            ax = axes[0, 1]
        else:
            ax = axes[1]

        ax.imshow(res_b_norm)

        if plot_masked:
            assert mask_a is not None
            assert mask_b is not None

            fig.set_figheight(10)
            fig.set_figwidth(15)

            D = np.shape(res_a)[2]
            mask_a_repeat = np.repeat(mask_a[:, :, np.newaxis], D, axis=2)
            mask_b_repeat = np.repeat(mask_b[:, :, np.newaxis], D, axis=2)
            res_a_mask = mask_a_repeat * res_a
            res_b_mask = mask_b_repeat * res_b

            if descriptor_image_stats is None:
                res_a_norm_mask, res_b_norm_mask = dc_plotting.normalize_masked_descriptor_pair(res_a, res_b, mask_a,
                                                                                                mask_b)
            else:
                res_a_norm_mask = dc_plotting.normalize_descriptor(res_a_mask, descriptor_image_stats['mask_image'])
                res_b_norm_mask = dc_plotting.normalize_descriptor(res_b_mask, descriptor_image_stats['mask_image'])

            res_a_norm_mask = res_a_norm_mask * mask_a_repeat
            res_b_norm_mask = res_b_norm_mask * mask_b_repeat

            axes[1, 0].imshow(res_a_norm_mask)
            axes[1, 1].imshow(res_b_norm_mask)

    @staticmethod
    def clip_pixel_to_image_size_and_round(uv, image_width, image_height):
        u = min(int(uv[0]), image_width - 1)
        v = min(int(uv[1]), image_height - 1)
        return (u, v)

    @staticmethod
    def single_same_scene_image_pair_quantitative_analysis(dcn, dataset, scene_name,
                                                           img_a_idx, img_b_idx,
                                                           num_matches=100,
                                                           debug=False):
        """
        Quantitative analysis of a dcn on a pair of images from the same scene.

        :param dcn: 
        :type dcn: DenseCorrespondenceNetwork
        :param dataset:
        :type dataset: SpartanDataset
        :param scene_name:
        :type scene_name: str
        :param img_a_idx:
        :type img_a_idx: int
        :param img_b_idx:
        :type img_b_idx: int
        :param camera_intrinsics_matrix: Optionally set camera intrinsics, otherwise will get it from the dataset
        :type camera_intrinsics_matrix: 3 x 3 numpy array
        :return: Dict with relevant data
        :rtype:
        """

        rgb_a, mask_a = dataset.get_rgb_mask(scene_name, img_a_idx)

        rgb_b, mask_b = dataset.get_rgb_mask('shirt_canonical', img_b_idx)

        knots_a = dataset.get_knots_info(scene_name)
        knots_b = dataset.get_knots_info('shirt_canonical')
        img_a_knots, img_b_knots = knots_a[str(img_a_idx)], knots_b[str(img_b_idx)]
        
        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        # find correspondences
        (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(img_a_knots, img_b_knots,
                                                                                      device='GPU')

        if uv_a_vec is None:
            print("no matches found, returning")
            return None

        # container to hold a list of pandas dataframe
        # will eventually combine them all with concat
        dataframe_list = []

        total_num_matches = len(uv_a_vec[0])
        num_matches = min(num_matches, total_num_matches)
        match_list = random.sample(range(0, total_num_matches), num_matches)

        if debug:
            match_list = [50]

        logging_rate = 100

        image_height, image_width = dcn.image_shape

        DCE = DenseCorrespondenceEvaluation

        for i in match_list:
            uv_a = (uv_a_vec[0][i], uv_a_vec[1][i])
            uv_b_raw = (uv_b_vec[0][i], uv_b_vec[1][i])
            uv_b = DCE.clip_pixel_to_image_size_and_round(uv_b_raw, image_width, image_height)

            pd_template = DCE.compute_descriptor_match_statistics(mask_a,
                                                                  mask_b,
                                                                  uv_a,
                                                                  uv_b,
                                                                  res_a,
                                                                  res_b,
                                                                  rgb_a=rgb_a,
                                                                  rgb_b=rgb_b,
                                                                  debug=debug)

            pd_template.set_value('scene_name', scene_name)
            pd_template.set_value('img_a_idx', img_a_idx)
            pd_template.set_value('img_b_idx', img_b_idx)

            dataframe_list.append(pd_template.dataframe)

        return dataframe_list

    @staticmethod
    def compute_descriptor_match_statistics(mask_a, mask_b, uv_a, uv_b,
                                            res_a, res_b, params=None,
                                            rgb_a=None, rgb_b=None, debug=False):
        """
        Computes statistics of descriptor pixelwise match.
        """
        
        height, width, _ = res_a.shape
        DCE = DenseCorrespondenceEvaluation
        uv_a = (min(width-1, uv_a[0]), min(height-1, uv_a[1]))
        uv_b_sym = (width - 1 - uv_b[0], uv_b[1])
        # compute best match
        uv_b_pred, best_match_diff, norm_diffs = \
            DenseCorrespondenceNetwork.find_best_match(uv_a, res_a,
                                                       res_b, debug=debug)

        # norm_diffs shape is (H,W)

        # compute best match on mask only
        mask_b_inv = 1 - mask_b
        masked_norm_diffs = norm_diffs + mask_b_inv * 1e6

        best_match_flattened_idx_masked = np.argmin(masked_norm_diffs)
        best_match_xy_masked = np.unravel_index(best_match_flattened_idx_masked, masked_norm_diffs.shape)
        best_match_diff_masked = masked_norm_diffs[best_match_xy_masked]
        uv_b_pred_masked = (best_match_xy_masked[1], best_match_xy_masked[0])

        # compute pixel space difference
#         pixel_match_error_l2 = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=2)
#         pixel_match_error_l2_masked = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred_masked)), ord=2)
#         pixel_match_error_l1 = np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=1)
        
        pixel_match_error_l2 = min(np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=2),
                                   np.linalg.norm((np.array(uv_b_sym) - np.array(uv_b_pred)), ord=2))
        pixel_match_error_l2_masked = min(np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred_masked)), ord=2),
                                          np.linalg.norm((np.array(uv_b_sym) - np.array(uv_b_pred_masked)), ord=2))
        pixel_match_error_l1 = min(np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=1),
                                   np.linalg.norm((np.array(uv_b_sym) - np.array(uv_b_pred)), ord=1))

        # extract the ground truth descriptors
        des_a = res_a[uv_a[1], uv_a[0], :]
        des_b_ground_truth = res_b[uv_b[1], uv_b[0], :]
        norm_diff_descriptor_ground_truth = np.linalg.norm(des_a - des_b_ground_truth)

        # from Schmidt et al 2017: 
        """
        We then determine the number of pixels in the target image that are closer in
        descriptor space to the source point than the manually-labelled corresponding point.
        """
        # compute this
        (v_indices_better_than_ground_truth, u_indices_better_than_ground_truth) = np.where(
            norm_diffs < norm_diff_descriptor_ground_truth)
        num_pixels_closer_than_ground_truth = len(u_indices_better_than_ground_truth)
        num_pixels_in_image = res_a.shape[0] * res_a.shape[1]
        fraction_pixels_closer_than_ground_truth = num_pixels_closer_than_ground_truth * 1.0 / num_pixels_in_image

        (v_indices_better_than_ground_truth_masked, u_indices_better_than_ground_truth_masked) = np.where(
            masked_norm_diffs < norm_diff_descriptor_ground_truth)
        num_pixels_closer_than_ground_truth_masked = len(u_indices_better_than_ground_truth_masked)
        num_pixels_in_masked_image = len(np.nonzero(mask_b)[0])
        fraction_pixels_closer_than_ground_truth_masked = num_pixels_closer_than_ground_truth_masked * 1.0 / num_pixels_in_masked_image

        # new metric: average l2 distance of the pixels better than ground truth
        if num_pixels_closer_than_ground_truth == 0:
            average_l2_distance_for_false_positives = 0.0
        else:
            l2_distances = np.sqrt((u_indices_better_than_ground_truth - uv_b[0]) ** 2 + (
                    v_indices_better_than_ground_truth - uv_b[1]) ** 2)
            average_l2_distance_for_false_positives = np.average(l2_distances)

        # new metric: average l2 distance of the pixels better than ground truth
        if num_pixels_closer_than_ground_truth_masked == 0:
            average_l2_distance_for_false_positives_masked = 0.0
        else:
            l2_distances_masked = np.sqrt((u_indices_better_than_ground_truth_masked - uv_b[0]) ** 2 + (
                    v_indices_better_than_ground_truth_masked - uv_b[1]) ** 2)
            average_l2_distance_for_false_positives_masked = np.average(l2_distances_masked)

        if debug:
            fig, axes = correspondence_plotter.plot_correspondences_direct(rgb_a, rgb_b,
                                                                           uv_a, uv_b, show=False)

            correspondence_plotter.plot_correspondences_direct(rgb_a, rgb_b,
                                                               uv_a, uv_b_pred,
                                                               use_previous_plot=(fig, axes),
                                                               show=True,
                                                               circ_color='purple')

        pd_template = DCNEvaluationPandaTemplate()
       
        pd_template.set_value('pixel_match_error_l2', pixel_match_error_l2)
        pd_template.set_value('pixel_match_error_l2_masked', pixel_match_error_l2_masked)
        pd_template.set_value('pixel_match_error_l1', pixel_match_error_l1)

        pd_template.set_value('fraction_pixels_closer_than_ground_truth', fraction_pixels_closer_than_ground_truth)
        pd_template.set_value('fraction_pixels_closer_than_ground_truth_masked',
                              fraction_pixels_closer_than_ground_truth_masked)
        pd_template.set_value('average_l2_distance_for_false_positives', average_l2_distance_for_false_positives)
        pd_template.set_value('average_l2_distance_for_false_positives_masked',
                              average_l2_distance_for_false_positives_masked)

        return pd_template

    @staticmethod
    def get_random_scenes_and_image_pairs(dataset, num_pairs=5):
        """
        Given a dataset, chose a variety of random scenes and image pairs

        :param dataset: dataset from which to draw a scene and image pairs
        :type dataset: SpartanDataset

        :return: scene_names, img_pairs
        :rtype: list[str], list of lists, where each of the lists are [img_a_idx, img_b_idx], for example:
            [[113,220],
             [114,225]]
        """

        scene_names = []

        img_pairs = []
        for _ in range(num_pairs):
            scene_name = dataset.get_random_scene_name()
            img_a_idx = dataset.get_random_image_index(scene_name)
            img_b_idx = dataset.get_random_image_index('shirt_canonical')
            img_pairs.append([img_a_idx, img_b_idx])
            scene_names.append(scene_name)

        return scene_names, img_pairs

    @staticmethod
    def compute_loss_on_dataset(dcn, data_loader, loss_config, num_iterations=500, ):
        """

        Computes the loss for the given number of iterations

        :param dcn:
        :type dcn:
        :param data_loader:
        :type data_loader:
        :param num_iterations:
        :type num_iterations:
        :return:
        :rtype:
        """
        dcn.eval()

        # loss_vec = np.zeros(num_iterations)
        loss_vec = []
        match_loss_vec = []
        non_match_loss_vec = []
        counter = 0
        pixelwise_contrastive_loss = PixelwiseContrastiveLoss(dcn.image_shape, config=loss_config)

        batch_size = 1

        for i, data in enumerate(data_loader, 0):

            # get the inputs
            data_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b, metadata = data
            data_type = data_type[0]

            if len(matches_a[0]) == 0:
                print("didn't have any matches, continuing")
                continue

            img_a = Variable(img_a.cuda(), requires_grad=False)
            img_b = Variable(img_b.cuda(), requires_grad=False)

            if data_type == "matches":
                matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
                non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)

            # run both images through the network
            image_a_pred = dcn.forward(img_a)
            image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

            image_b_pred = dcn.forward(img_b)
            image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

            # get loss
            if data_type == "matches":
                loss, match_loss, non_match_loss = \
                    pixelwise_contrastive_loss.get_loss(image_a_pred,
                                                        image_b_pred,
                                                        matches_a,
                                                        matches_b,
                                                        non_matches_a,
                                                        non_matches_b)

                loss_vec.append(loss.data[0])
                non_match_loss_vec.append(non_match_loss.data[0])
                match_loss_vec.append(match_loss.data[0])

            if i > num_iterations:
                break

        loss_vec = np.array(loss_vec)
        match_loss_vec = np.array(match_loss_vec)
        non_match_loss_vec = np.array(non_match_loss_vec)

        loss = np.average(loss_vec)
        match_loss = np.average(match_loss_vec)
        non_match_loss = np.average(non_match_loss_vec)

        return loss, match_loss, non_match_loss

    @staticmethod
    def compute_descriptor_statistics_on_dataset(dcn, dataset, num_images=100,
                                                 save_to_file=True, filename=None):
        """
        Computes the statistics of the descriptors on the dataset
        :param dcn:
        :type dcn:
        :param dataset:
        :type dataset:
        :param save_to_file:
        :type save_to_file:
        :return:
        :rtype:
        """

        utils.reset_random_seed()
        print("doing dcn eval")
        dcn.eval()
        to_tensor = transforms.ToTensor()

        # compute the per-channel mean
        def compute_descriptor_statistics(res, mask_tensor):
            """
            Computes
            :param res: The output of the DCN
            :type res: torch.FloatTensor with shape [H,W,D]
            :return: min, max, mean
            :rtype: each is torch.FloatTensor of shape [D]
            """
            # convert to [W*H, D]
            D = res.shape[2]

            # convert to torch.FloatTensor instead of variable
            if isinstance(res, torch.autograd.Variable):
                res = res.data

            res_reshape = res.contiguous().view(-1, D)
            channel_mean = res_reshape.mean(0)  # shape [D]
            channel_min, _ = res_reshape.min(0)  # shape [D]
            channel_max, _ = res_reshape.max(0)  # shape [D]

            mask_flat = mask_tensor.view(-1, 1).squeeze(1)

            # now do the same for the masked image
            # gracefully handle the case where the mask is all zeros
            mask_indices_flat = torch.nonzero(mask_flat)
            if len(mask_indices_flat) == 0:
                return None, None

            mask_indices_flat = mask_indices_flat.squeeze(1)
            mask_indices_flat = mask_indices_flat.to('cpu')

            # print "mask_flat.shape", mask_flat.shape

            res_masked_flat = res_reshape.index_select(0, mask_indices_flat)  # shape [mask_size, D]
            mask_channel_mean = res_masked_flat.mean(0)
            mask_channel_min, _ = res_masked_flat.min(0)
            mask_channel_max, _ = res_masked_flat.max(0)

            entire_image_stats = (channel_min, channel_max, channel_mean)
            mask_image_stats = (mask_channel_min, mask_channel_max, mask_channel_mean)
            return entire_image_stats, mask_image_stats

        def compute_descriptor_std_dev(res, channel_mean):
            """
            Computes the std deviation of a descriptor image, given a channel mean
            :param res:
            :type res:
            :param channel_mean:
            :type channel_mean:
            :return:
            :rtype:
            """
            D = res.shape[2]
            res_reshape = res.view(-1, D)  # shape [W*H,D]
            v = res - channel_mean
            std_dev = torch.std(v, 0)  # shape [D]
            return std_dev

        def update_stats(stats_dict, single_img_stats):
            """
            Update the running mean, min and max
            :param stats_dict:
            :type stats_dict:
            :param single_img_stats:
            :type single_img_stats:
            :return:
            :rtype:
            """

            min_temp, max_temp, mean_temp = single_img_stats

            if stats_dict['min'] is None:
                stats_dict['min'] = min_temp
            else:
                stats_dict['min'] = torch.min(stats_dict['min'], min_temp)

            if stats_dict['max'] is None:
                stats_dict['max'] = max_temp
            else:
                stats_dict['max'] = torch.max(stats_dict['max'], max_temp)

            if stats_dict['mean'] is None:
                stats_dict['mean'] = mean_temp
            else:
                stats_dict['mean'] += mean_temp

        stats = dict()
        stats['entire_image'] = {'mean': None, 'max': None, 'min': None}
        stats['mask_image'] = {'mean': None, 'max': None, 'min': None}

        for _ in range(0, num_images):
            rgb, mask = dataset.get_random_rgb_mask()
            img_tensor = dataset.rgb_image_to_tensor(rgb)
            res = dcn.forward_single_image_tensor(img_tensor)  # [H, W, D]

            mask_tensor = to_tensor(mask).cuda()
            entire_image_stats, mask_image_stats = compute_descriptor_statistics(res, mask_tensor)

            # handles the case of an empty mask
            if mask_image_stats is None:
                logging.info("Mask was empty, skipping")
                continue

            update_stats(stats['entire_image'], entire_image_stats)
            update_stats(stats['mask_image'], mask_image_stats)

        for key, val in stats.items():
            val['mean'] = 1.0 / num_images * val['mean']
            for field in val:
                val[field] = val[field].tolist()

        if save_to_file:
            if filename is None:
                path_to_params_folder = dcn.config['path_to_network_params_folder']
                path_to_params_folder = utils.convert_to_absolute_path(path_to_params_folder)
                filename = os.path.join(path_to_params_folder, 'descriptor_statistics.yaml')

            utils.saveToYaml(stats, filename)

        return stats

    @staticmethod
    def run_evaluation_on_network(model_folder, num_image_pairs=100,
                                  num_matches_per_image_pair=100,
                                  save_folder_name="analysis",
                                  compute_descriptor_statistics=True,
                                  cross_scene=False,
                                  dataset=None):
        """
        Runs all the quantitative evaluations on the model folder
        Creates a folder model_folder/analysis that stores the information.

        Performs several steps:

        1. compute dataset descriptor stats
        2. compute quantitative eval csv files
        3. make quantitative plots, save as a png for easy viewing


        :param model_folder:
        :type model_folder:
        :return:
        :rtype:
        """

        utils.reset_random_seed()

        DCE = DenseCorrespondenceEvaluation

        model_folder = utils.convert_to_absolute_path(model_folder)
        print("got", model_folder)

        # save it to a csv file
        output_dir = os.path.join(model_folder, save_folder_name)
        train_output_dir = os.path.join(output_dir, "train")
        test_output_dir = os.path.join(output_dir, "test")
        cross_scene_output_dir = os.path.join(output_dir, "cross_scene")
        print("output_dir: ", output_dir)
        print("train_output_dir: ", train_output_dir)
        print("test_output_dir: ", test_output_dir)
        print("cross_scene_output_dir:", cross_scene_output_dir)

        # create the necessary directories
        print("\ncreating necessary dirs")
        for dir in [output_dir, train_output_dir, test_output_dir, cross_scene_output_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)

        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder)
        dcn.eval()

        if dataset is None:
            dataset = dcn.load_training_dataset()

        # compute dataset statistics
        if compute_descriptor_statistics:
            logging.info("Computing descriptor statistics on dataset")
            print("computing stats")
            DCE.compute_descriptor_statistics_on_dataset(dcn, dataset, num_images=100, save_to_file=True)
            print("done computing stats")

        # evaluate on training data and on test data
        logging.info("Evaluating network on train data")
        dataset.set_train_mode()
        pd_dataframe_list, df = DCE.evaluate_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        train_csv = os.path.join(train_output_dir, "data.csv")
        df.to_csv(train_csv)

        logging.info("Evaluating network on test data")
        dataset.set_test_mode()
        pd_dataframe_list, df = DCE.evaluate_network(dcn, dataset, num_image_pairs=num_image_pairs,
                                                     num_matches_per_image_pair=num_matches_per_image_pair)

        test_csv = os.path.join(test_output_dir, "data.csv")
        df.to_csv(test_csv)

        if cross_scene:
            logging.info("Evaluating network on cross scene data")
            df = DCE.evaluate_network_cross_scene(dcn=dcn, dataset=dataset, save=False)
            cross_scene_csv = os.path.join(cross_scene_output_dir, "data.csv")
            df.to_csv(cross_scene_csv)

        logging.info("Making plots")
        DCEP = DenseCorrespondenceEvaluationPlotter
        fig_axes = DCEP.run_on_single_dataframe(train_csv, label="train", save=False)
        fig_axes = DCEP.run_on_single_dataframe(test_csv, label="test", save=False, previous_fig_axes=fig_axes)
        if cross_scene:
            fig_axes = DCEP.run_on_single_dataframe(cross_scene_csv, label="cross_scene", save=False,
                                                    previous_fig_axes=fig_axes)

        fig, _ = fig_axes
        save_fig_file = os.path.join(output_dir, "quant_plots.png")
        fig.savefig(save_fig_file)

        # only do across object analysis if have multiple single objects
        if dataset.get_number_of_unique_single_objects() > 1:
            across_object_output_dir = os.path.join(output_dir, "across_object")
            if not os.path.isdir(across_object_output_dir):
                os.makedirs(across_object_output_dir)
            logging.info("Evaluating network on across object data")
            df = DCE.evaluate_network_across_objects(dcn=dcn, dataset=dataset)
            across_object_csv = os.path.join(across_object_output_dir, "data.csv")
            df.to_csv(across_object_csv)
            DCEP.run_on_single_dataframe_across_objects(across_object_csv, label="across_object", save=True)

        logging.info("Finished running evaluation on network")

    @staticmethod
    def evaluate_network_by_zones(dcn, dataset, num_images=100, use_symmetry=False, num_matches=2000, sigma=3):

        dcn.eval()        

        scene_name = dataset.get_random_scene_name()
        scene_name_ref = 'shirt_canonical'
        ref_idx = dataset.get_random_image_index('shirt_canonical')

        # Get reference image
        rgb_ref, _, mask_ref, _ = dataset.get_rgbd_mask_pose(scene_name_ref, ref_idx)
        rgb_ref, mask_ref = (np.asarray(rgb_ref), np.asarray(mask_ref))
        rgb_ref_tensor = dataset.rgb_image_to_tensor(rgb_ref)
        res_ref = dcn.forward_single_image_tensor(rgb_ref_tensor).data.cpu().numpy()

        img_width, img_height = mask_ref.shape
        
        # Get the knots for the canonical and evaluation datasets
        knots = dataset.get_knots_info(scene_name)
        knots_canonical = dataset.get_knots_info(scene_name_ref)
        
        # Matrix of results
        img_dists = np.zeros(rgb_ref.shape[:2])
        counts = np.ones(rgb_ref.shape[:2])

        img_idxs = random.sample(list(knots), num_images)
        for i, img_idx in enumerate(img_idxs):
            print("Computed statistics on {}/{} images".format(i, len(img_idxs)), end='\r')
            
            img_knots = knots[img_idx]
            ref_img_knots = knots_canonical[ref_idx]

            # Get evaluation image
            rgb_img, mask_img = dataset.get_rgb_mask(scene_name, img_idx) 
            rgb_img_tensor = dataset.rgb_image_to_tensor(rgb_img)
            res_img = dcn.forward_single_image_tensor(rgb_img_tensor).data.cpu().numpy()
            
            (uv_img_vec, uv_ref_vec) = correspondence_finder.batch_find_pixel_correspondences(
                img_knots, ref_img_knots, device='GPU')

            total_num_matches = len(uv_img_vec[0])
            num_matches = min(num_matches, total_num_matches)
            match_list = random.sample(range(0, total_num_matches), num_matches)

            for j in match_list:
                uv_img = (uv_img_vec[0][j], uv_img_vec[1][j])
                uv_ref = (uv_ref_vec[0][j].cpu(), uv_ref_vec[1][j].cpu())
                uv_ref_sym = (img_width - 1 - uv_ref[0], uv_ref[1])

                uv_ref_pred, _, _ = \
                    DenseCorrespondenceNetwork.find_best_match(uv_img, res_img, res_ref, mask_b=mask_img)
                
                pixel_match_error_l2 = min(np.linalg.norm((np.array(uv_ref) - np.array(uv_ref_pred)), ord=2),
                                           np.linalg.norm((np.array(uv_ref_sym) - np.array(uv_ref_pred)), ord=2))
                
                img_dists[int(uv_ref[1]), int(uv_ref[0])] += pixel_match_error_l2
                counts[int(uv_ref[1]), int(uv_ref[0])] += 1

        img_dists_norm = np.divide(img_dists, counts)
#         img_dists_norm = dc_plotting.normalize_vec(img_dists_norm)
        img_dists_norm = gaussian_filter(img_dists_norm, sigma=sigma)
        img_dists_norm *= mask_ref
        
        plt.imshow(img_dists_norm, cmap='jet')
        plt.colorbar()
        plt.title('Heatmap of pixel distance to the best match')
    
    
    @staticmethod
    def evaluate_network_qualitative(dcn, dataset, real_data=False, type_of_analysis='single_match', num_pairs=5, shirt='shirt1', idx_real='27'):

        # type_of_analysis = ['general', 'single_match', 'circular', 'iterative', 'inverse_prediction']
        dcn.eval()
        scene_names, img_pairs = DenseCorrespondenceEvaluation.get_random_scenes_and_image_pairs(dataset, num_pairs)
        for scene_name, img_pair in zip(scene_names, img_pairs):
            print("Image pair (%s, %s)" % (img_pair[0], img_pair[1]))
            if real_data:
                scene_name = '0120_' + shirt
                img_pair[0] = idx_real
            data = DenseCorrespondenceEvaluation.get_data(dcn, dataset, scene_name, img_pair[0], img_pair[1], real_data=real_data)
            if type_of_analysis == 'single_match':
                DenseCorrespondenceEvaluation.single_match_qualitative_analysis(data, pca_plot=True)
            elif type_of_analysis == 'general':
                DenseCorrespondenceEvaluation.image_pair_qualitative_analysis(dcn, data)
            elif type_of_analysis == 'circular':
                DenseCorrespondenceEvaluation.circular_qualitative_analysis(data)
            elif type_of_analysis == 'iterative':
                DenseCorrespondenceEvaluation.iterative_qualitative_analysis(data)
            elif type_of_analysis == 'inverse_prediction':
#                 DenseCorrespondenceEvaluation.inverse_prediction_analysis(data)
                data = DenseCorrespondenceEvaluation.get_data(dcn, dataset, scene_name, img_pair[0], img_pair[1], real_data=real_data, inverse=True)
                DenseCorrespondenceEvaluation.single_match_qualitative_analysis(data, pca_plot=True)


    @staticmethod
    def draw_a_circle(img_array, pixel_center, radius=14, color=COLOR_BLUE):
        if pixel_center is not None:
            img_array = cv2.circle(img_array, pixel_center, radius=radius, color=COLOR_WHITE, thickness=-1)
            img_array = cv2.circle(img_array, pixel_center, radius=radius-4, color=color, thickness=-1)
        return img_array

    @staticmethod
    def get_data(dcn, dataset, scene_name, img_a_idx, img_b_idx, real_data=False, inverse=False):

        rgb_a, mask_a = dataset.get_rgb_mask(scene_name, img_a_idx)
        rgb_b, mask_b = dataset.get_rgb_mask('shirt_canonical', img_b_idx)

        rgb_a_array = np.array(rgb_a)
        rgb_b_array = np.array(rgb_b)

        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu()
        
        # get real matches
        if real_data:
            sampled_idx_list = random_sample_from_masked_image(mask_a, 1)
            pixel_a = (sampled_idx_list[1][0], sampled_idx_list[0][0])
            pixel_b = None
            if inverse:
                sampled_idx_list = random_sample_from_masked_image(mask_b, 1)
                pixel_b = (sampled_idx_list[1][0], sampled_idx_list[0][0])
                pixel_a = None
        else:
            knots_a = dataset.get_knots_info(scene_name)
            knots_b = dataset.get_knots_info('shirt_canonical')
            img_a_knots, img_b_knots = knots_a[str(img_a_idx)], knots_b[str(img_b_idx)]

            (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(img_a_knots, img_b_knots)
            total_matches = len(uv_a_vec[0])
            if total_matches == 0:
                return rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, (0,0), (0,0), dcn.config['loss_type']
            random_point = random.choice(range(total_matches))
            pixel_a = (int(uv_a_vec[0][random_point]), int(uv_a_vec[1][random_point]))
            pixel_b = (int(uv_b_vec[0][random_point]), int(uv_b_vec[1][random_point]))
        
        if inverse:
            return rgb_b_array, rgb_a_array, rgb_b_tensor, rgb_a_tensor, mask_b, mask_a, res_b, res_a, pixel_b, pixel_a, dcn.config['loss_type']
        return rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, dcn.config['loss_type']

    @staticmethod
    def image_pair_qualitative_analysis(dcn, data, num_matches=7):

        rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, loss_type = data

        # sample points on img_a. Compute best matches on img_b
        # note that this is in (x,y) format
        # TODO: if this mask is empty, this function will not be happy
        # de-prioritizing since this is only for qualitative evaluation plots
        sampled_idx_list = random_sample_from_masked_image(mask_a, num_matches)

        # list of cv2.KeyPoint
        kp1 = []
        kp2 = []
        matches = []  # list of cv2.DMatch

        # placeholder constants for opencv
        diam = 0.01
        dist = 0.01

        try:
            descriptor_image_stats = dcn.descriptor_image_stats
        except:
            print("Could not find descriptor image stats...")
            print("Only normalizing pairs of images!")
            descriptor_image_stats = None

        for i in range(0, num_matches):
            # convert to (u,v) format
            pixel_a = [sampled_idx_list[1][i], sampled_idx_list[0][i]]
            best_match_uv, _, _ = \
                DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a, res_b)

            # be careful, OpenCV format is  (u,v) = (right, down)
            kp1.append(cv2.KeyPoint(float(pixel_a[0]), float(pixel_a[1]), diam))
            kp2.append(cv2.KeyPoint(float(best_match_uv[0]), float(best_match_uv[1]), diam))
            matches.append(cv2.DMatch(i, i, dist))

        gray_a_numpy = cv2.cvtColor(rgb_a_array, cv2.COLOR_BGR2GRAY)
        gray_b_numpy = cv2.cvtColor(rgb_b_array, cv2.COLOR_BGR2GRAY)
        img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, matches, flags=2, outImg=gray_b_numpy)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        axes.imshow(img3)

        # show colormap if possible (i.e. if descriptor dimension is 1 or 3)
        if dcn.descriptor_dimension in [1, 3]:
            DenseCorrespondenceEvaluation.plot_descriptor_colormaps(res_a, res_b,
                                                                    descriptor_image_stats=descriptor_image_stats,
                                                                    mask_a=mask_a,
                                                                    mask_b=mask_b,
                                                                    plot_masked=True)

        plt.show()
  
    @staticmethod
    def single_match_qualitative_analysis(data, pca_plot=False):

        rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, loss_type = data

        fig, axes = plt.subplots(nrows=2 if pca_plot else 1, ncols=3, figsize=(15,7))
        
        rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, pixel_a, color=COLOR_GREEN)
        axes[0,0].imshow(rgb_a_array)
        axes[0,0].set_title('Img A: Queried pixel')

        uv_b, _, norm_diffs = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a.numpy(), res_b.numpy())
        uv_b_masked, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b)
        
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, uv_b, color=COLOR_RED)
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, uv_b_masked, color=COLOR_BLUE)
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, pixel_b, color=COLOR_GREEN)
        axes[0,1].imshow(rgb_b_array)
        axes[0,1].set_title('Img B: Best match')
        
        # heatmap
        if loss_type == 'distributional':
            norm_diffs = torch.tensor(norm_diffs)
            norm_diffs = norm_diffs.ravel()
            p_b = F.softmax(-1 * norm_diffs, dim=0).double() # compute current distribution   
            p_b = p_b.reshape(rgb_b_tensor.shape[1:]).numpy()   
            # p_b += np.fliplr(p_b)
            
            data_sorted = np.sort(p_b.ravel())
            p = 1. * np.arange(len(p_b.ravel())) / (len(p_b.ravel()) - 1)
            axes[1,2].plot(p, data_sorted)
            axes[1,2].set_ylabel('Probability across pixels')
            axes[1,2].set_xlabel('Percentage of image pixels')
            
            im = axes[0,2].imshow(p_b, cmap='jet', alpha=0.95)#, vmin=0, vmax=0.005)
            im2 = axes[0,2].imshow(rgb_b_array, alpha=0.15)
            plt.colorbar(im, ax=axes[0,2], fraction=0.046, pad=0.04)
            #axes[0,2].set_xlim([0,320])
            axes[0,2].set_title('Img B: Best match heatmap')
        
        elif loss_type == 'contrastive':
            norm_diffs = (norm_diffs - np.min(norm_diffs)) / (np.max(norm_diffs) - np.min(norm_diffs))
            norm_diffs = 1 - norm_diffs
            im = axes[1,0].imshow(norm_diffs, cmap='jet')
            fig.colorbar(im, ax=axes[1,0])
            axes[1,0].set_title('Best match heatmap')
            
            im = axes[1,1].imshow(norm_diffs*mask_b, cmap='jet')
            fig.colorbar(im, ax=axes[1,1])
            axes[1,1].set_title('Best match heatmap masked')
        
        # Compute PCA of the canonical image
        if pca_plot:
            d = res_a.shape[-1]
            pca = PCA(n_components=3)
            res_b_pc = pca.fit_transform(res_b.reshape((-1, d))).reshape(res_b.shape[0], res_b.shape[1], -1)
            res_a_pc = pca.transform(res_a.reshape((-1, d))).reshape(res_a.shape[0], res_a.shape[1], -1)

            res_a_pc = dc_plotting.normalize_descriptor(res_a_pc)
            res_a_pc *= np.repeat(mask_a[:, :, np.newaxis], 3, axis=2)

            res_b_pc = dc_plotting.normalize_descriptor(res_b_pc)
            res_b_pc *= np.repeat(mask_b[:, :, np.newaxis], 3, axis=2)
            
            axes[1,0].imshow(res_a_pc)
            axes[1,0].set_title('Img A: PCA plot DOD')

            axes[1,1].imshow(res_b_pc)
            axes[1,1].set_title('Img B: PCA plot DOD')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def circular_qualitative_analysis(data):

        rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, loss_type = data

        noise = np.random.multivariate_normal([0,0], [[100,0],[0,100]], 7)
        pixels_around_a = [(pixel_a[0] + int(x), pixel_a[1] + int(y)) for x,y in noise]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, pixel_a, color=COLOR_GREEN)
        for pixel_a_random in pixels_around_a:
            rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, pixel_a_random, color=COLOR_RED, radius=5)
        axes[0].imshow(rgb_a_array)
        axes[0].set_title('Queried pixel')

        uv_b_masked, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b)
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, uv_b_masked, color=COLOR_BLUE)
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, pixel_b, color=COLOR_GREEN)
        for pixel_a_random in pixels_around_a:
            uv_b_masked_random, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_a_random, res_a.numpy(), res_b.numpy(), mask_b=mask_b)
            rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, uv_b_masked_random, color=COLOR_RED, radius=5)
        axes[1].imshow(rgb_b_array)
        axes[1].set_title('Best match')
    
    @staticmethod
    def iterative_qualitative_analysis(data, num_iters=8):

        rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, loss_type = data

        ncols = (num_iters+1)//2
        fig, axes = plt.subplots(nrows=2, ncols=ncols)
        fig.set_figheight(5*2)
        fig.set_figwidth(7.5*ncols)
        
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, pixel_b, color=COLOR_BLUE)
        
        axes[0,0].imshow(DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, pixel_a, color=COLOR_BLUE))
        axes[0,0].set_title('Queried pixel')
            
        for iter in range(1, num_iters):
            uv_b, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b)

            axes[iter//ncols, iter%ncols].imshow(DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, uv_b, color=COLOR_RED))
            axes[iter//ncols, iter%ncols].set_title('Best match')

            rgb_a_array, rgb_b_array = rgb_b_array, rgb_a_array
            mask_a, mask_b = mask_b, mask_a
            res_a, res_b = res_b, res_a
            pixel_a = uv_b  
        
    @staticmethod
    def inverse_prediction_analysis(data):

        rgb_a_array, rgb_b_array, rgb_a_tensor, rgb_b_tensor, mask_a, mask_b, res_a, res_b, pixel_a, pixel_b, loss_type = data

#         sampled_idx_list = random_sample_from_masked_image(mask_a, 100)
#         pixels_a = [(u,v) for u,v in zip(sampled_idx_list[1], sampled_idx_list[0])]
        
        pixel_a = None
        sampled_idx_list = random_sample_from_masked_image(mask_b, 1)
        pixel_b = (sampled_idx_list[1][0], sampled_idx_list[0][0])

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        
        rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, pixel_a, color=COLOR_GREEN)
        rgb_b_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_b_array, pixel_b, color=COLOR_GREEN)        

#         min_dist = 1e5
#         closest_pixel = None
#         for pixel_a in pixels_a:
#             uv_b_masked, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a.numpy(), res_b.numpy(), mask_b=mask_b)
#             dist = np.linalg.norm((np.array(uv_b_masked) - np.array(pixel_b)), ord=2)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_pixel = pixel_a
        
        axes[0].imshow(rgb_b_array)
        axes[0].set_title('Queried pixel')
        
#         rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, closest_pixel, color=COLOR_RED)
        closest_pixel_direct, _, _ = DenseCorrespondenceNetwork.find_best_match(pixel_b, res_b.numpy(), res_a.numpy(), mask_b=mask_a)
        rgb_a_array = DenseCorrespondenceEvaluation.draw_a_circle(rgb_a_array, closest_pixel_direct, color=COLOR_BLUE)
        axes[1].imshow(rgb_a_array)
        axes[1].set_title('Best match')
    




    @staticmethod
    def evaluate_match_quality(dcn, dataset, num_image_pairs=100, logging_rate=5, kl_loss=False):

        DCE = DenseCorrespondenceEvaluation
        match_distances = []
        match_probabilities = []
        
        for i in range(0, num_image_pairs):
            scene_name = dataset.get_random_scene_name()
            idx_pair = DCE.get_image_pair_random(dataset, scene_name)
            img_idx_a, img_idx_b = idx_pair
            DCE.single_image_match_quality(dcn, dataset, scene_name, img_idx_a, img_idx_b, match_distances, match_probabilities, kl_loss=kl_loss)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(5)
        fig.set_figwidth(7)
        ax.scatter(match_distances, match_probabilities, s=0.5)
        ax.set_xlabel('Match distance')
        if not kl_loss:
            ax.set_ylabel('Match probability')
        else:
            ax.set_ylabel('KL loss')
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
        y = [int(i < 100) for i in match_distances]
        y_confidence = [-i for i in match_probabilities]
        fpr, tpr, thresholds = roc_curve(y, y_confidence, pos_label=1)
        auc_score = roc_auc_score(y, y_confidence)
        ax[0].plot(fpr, tpr)
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('ROC curve plot: AUC %.2f' % auc_score)

        best_thresh = 8.6
        y_pred = [int(i < best_thresh) for i in match_probabilities]
        cf_matrix = confusion_matrix(y, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=["Non match", "Match"])
        cm_display.plot(ax=ax[1], cmap="inferno")
        plt.tight_layout()
            
    
    @staticmethod
    def single_image_match_quality(dcn, dataset, scene_name, img_a_idx, img_b_idx, match_dists, match_probabilities, num_matches=25, kl_loss=False):

        rgb_a, mask_a = dataset.get_rgb_mask(scene_name, img_a_idx)
        rgb_b, mask_b = dataset.get_rgb_mask('shirt_canonical', img_b_idx)

        knots_a = dataset.get_knots_info(scene_name)
        knots_b = dataset.get_knots_info('shirt_canonical')
        
        img_a_knots, img_b_knots = knots_a[str(img_a_idx)], knots_b[str(img_b_idx)]
        
        mask_a = np.array(mask_a)
        mask_b = np.array(mask_b)

        # compute dense descriptors
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()

        # find correspondences
        (uv_a_vec, uv_b_vec) = correspondence_finder.batch_find_pixel_correspondences(img_a_knots, img_b_knots)

        if uv_a_vec is None:
            print("no matches found, returning")
            return None

        total_num_matches = len(uv_a_vec[0])
        num_matches = min(num_matches, total_num_matches)
        match_list = random.sample(range(0, total_num_matches), num_matches)

        image_height, image_width = dcn.image_shape

        for i in match_list:
            uv_a = (uv_a_vec[0][i], uv_a_vec[1][i])
            uv_b = (uv_b_vec[0][i], uv_b_vec[1][i])
            uv_b_sym = (image_width - 1 - uv_b[0], uv_b[1])

            # compute best match
            uv_b_pred, _, norm_diffs = \
                DenseCorrespondenceNetwork.find_best_match(uv_a, res_a, res_b, mask_b=mask_b)
            
            pixel_match_error_l2 = min(np.linalg.norm((np.array(uv_b) - np.array(uv_b_pred)), ord=2),
                                       np.linalg.norm((np.array(uv_b_sym) - np.array(uv_b_pred)), ord=2))
            match_dists.append(pixel_match_error_l2)
            
            # compute probability distribution
            norm_diffs = torch.tensor(norm_diffs)
            norm_diffs = norm_diffs.ravel()
            p_b = F.softmax(-1 * norm_diffs, dim=0).double() # compute current distribution   
            p_b = p_b.reshape(rgb_b_tensor.shape[1:]).numpy()          
            u, v = uv_b_pred       
            
            if not kl_loss:
                p_b += np.fliplr(p_b)
                radius = 2
                match_prob = 0
                for u_i, v_i in itertools.product(range(u-radius, u+radius+1), range(v-radius, v+radius+1)):
                    if (u_i-u)*(u_i-u) + (v_i-v)*(v_i-v) <= radius:
                        match_prob += p_b[u][v]
                match_probabilities.append(match_prob)
            
            else:
                pdl = PixelwiseDistributionalLoss(dcn.image_shape)
                masked_indices = pdl.flattened_mask_indices(torch.tensor(mask_b), inverse=True)
                q_b = pdl.gauss_2d_distribution(image_width, image_height, 1, u, v, masked_indices=masked_indices)
                loss = F.kl_div(torch.tensor(p_b.ravel()).cuda().log(), q_b, reduction='sum', log_target=False)
                match_probabilities.append(loss.item())
                
                

    @staticmethod
    def evaluate_distribution(dcn, dataset, masking=False, pca_components=3):

        dcn.eval()
        scene_name = dataset.get_random_scene_name()
        img_idx = dataset.get_random_image_index('shirt_canonical')
        rgb_a, _, mask_a, _ = dataset.get_rgbd_mask_pose('shirt_canonical', img_idx)

        mask_a = np.asarray(mask_a)
        rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu()
        
        d = res_a.shape[-1]
        pca = PCA(n_components=3)
        res_a_pc = pca.fit_transform(res_a.reshape((-1, d)))
        
        
        zones = np.zeros(mask_a.shape)
        for i, j in itertools.product(range(zones.shape[0]), range(zones.shape[1])):
            if mask_a[i][j] == 0:
                zones[i][j] = 0
            elif j < 270:
                zones[i][j] = 1
            elif j > 690:
                zones[i][j] = 2
            elif i > 300:
                zones[i][j] = 3
            else:
                zones[i][j] = 4

        translate = {
            0:'background',
            1:'left arm',
            2:'right arm',
            3:'upper torso',
            4:'lower torso'
        }
        
        colors = sns.color_palette()
        df = pd.DataFrame({
            'pc1': res_a_pc[:,0],
            'pc2': res_a_pc[:,1],
            'pc3': res_a_pc[:,2],
            'color': [colors[int(x)] for x in zones.ravel()],
            'Shirt zone': [translate[x] for x in zones.ravel()]
        })
        if masking:
            df = df.loc[df['Shirt zone'] != 'background']
            
        if pca_components == 2:
            fig, ax = plt.subplots(1,2, figsize=(15,6))
            ax[0].imshow(rgb_a)
            for z in df['Shirt zone'].unique():
                df_aux = df.loc[df['Shirt zone'] == z]
                ax[1].scatter(x=df_aux['pc1'], y=df_aux['pc2'], c=df_aux['color'], s=2, label=z)
            ax[1].set_xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 2)))
            ax[1].set_ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 2)))
            ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Shirt zone')
            ax[1].set_title('PCA of object descriptors by shirt zone')
            plt.show()
        elif pca_components == 3:
            plotly.offline.init_notebook_mode()
            fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3', color='Shirt zone',
                                labels={'pc1': 'PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 2)),
                                        'pc2': 'PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 2)),
                                        'pc3': 'PC3 ({}%)'.format(round(pca.explained_variance_ratio_[2], 2))})
            fig.update_traces(marker_size = 2)
            fig.show()




# -





class DenseCorrespondenceEvaluationPlotter(object):
    """
    This class contains plotting utilities. They are all
    encapsulated as static methods

    """

    def __init__(self):
        pass

    @staticmethod
    def make_cdf_plot(ax, data, num_bins, label=None, x_axis_scale_factor=1):
        """
        Plots the empirical CDF of the data
        :param ax: axis of a matplotlib plot to plot on
        :param data:
        :type data:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        cumhist, l, b, e = ss.cumfreq(data, num_bins)
        cumhist *= 1.0 / len(data)
        x_axis = l + b * np.arange(0, num_bins)
        x_axis /= x_axis_scale_factor
        plot = ax.plot(x_axis, cumhist, label=label)
        return plot

    @staticmethod
    def make_pixel_match_error_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df: pandas dataframe, i.e. generated from quantitative 
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        if masked:
            data_string = 'pixel_match_error_l2_masked'
        else:
            data_string = 'pixel_match_error_l2'

        data = df[data_string]

        # rescales the pixel distance to be relative to the diagonal of the image
        x_axis_scale_factor = 800

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label, x_axis_scale_factor=x_axis_scale_factor)
        if masked:
            ax.set_xlabel('Pixel match error (masked), L2 (pixel distance)')
        else:
            ax.set_xlabel('Pixel match error (fraction of image), L2 (pixel distance)')
        ax.set_ylabel('Fraction of images')

        return plot

    @staticmethod
    def make_across_object_best_match_plot(ax, df, label=None, num_bins=100):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df: pandas dataframe, i.e. generated from quantitative 
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['norm_diff_descriptor_best_match']

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Best descriptor match, L2 norm')
        ax.set_ylabel('Fraction of pixel samples from images')
        return plot

    @staticmethod
    def make_descriptor_accuracy_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        Makes a plot of best match accuracy.
        Drops nans
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        if masked:
            data_string = 'norm_diff_pred_3d_masked'
        else:
            data_string = 'norm_diff_pred_3d'

        data = df[data_string]
        data = data.dropna()
        data *= 100  # convert to cm

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        if masked:
            ax.set_xlabel('3D match error (masked), L2 (cm)')
        else:
            ax.set_xlabel('3D match error, L2 (cm)')
        ax.set_ylabel('Fraction of images')
        # ax.set_title("3D Norm Diff Best Match")
        return plot

    @staticmethod
    def make_norm_diff_ground_truth_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        data = df['norm_diff_descriptor_ground_truth']

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        ax.set_xlabel('Descriptor match error, L2')
        ax.set_ylabel('Fraction of images')
        return plot

    @staticmethod
    def make_fraction_false_positives_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        if masked:
            data_string = 'fraction_pixels_closer_than_ground_truth_masked'
        else:
            data_string = 'fraction_pixels_closer_than_ground_truth'

        data = df[data_string]

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)

        if masked:
            ax.set_xlabel('Fraction false positives (masked)')
        else:
            ax.set_xlabel('Fraction false positives')

        ax.set_ylabel('Fraction of images')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        return plot

    @staticmethod
    def make_average_l2_false_positives_plot(ax, df, label=None, num_bins=100, masked=False):
        """
        :param ax: axis of a matplotlib plot to plot on
        :param df:
        :type df:
        :param num_bins:
        :type num_bins:
        :return:
        :rtype:
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        if masked:
            data_string = 'average_l2_distance_for_false_positives_masked'
        else:
            data_string = 'average_l2_distance_for_false_positives'

        data = df[data_string]

        plot = DCEP.make_cdf_plot(ax, data, num_bins=num_bins, label=label)
        if masked:
            ax.set_xlabel('Average l2 pixel distance for false positives (masked)')
        else:
            ax.set_xlabel('Average l2 pixel distance for false positives')
        ax.set_ylabel('Fraction of images')
        # ax.set_xlim([0,200])
        return plot

    @staticmethod
    def compute_area_above_curve(df, field, num_bins=100):
        """
        Computes AOC for the entries in that field
        :param df:
        :type df: Pandas.DataFrame
        :param field: specifies which column of the DataFrame to use
        :type field: str
        :return:
        :rtype:
        """

        data = df[field]
        data = data.dropna()

        cumhist, l, b, e = ss.cumfreq(data, num_bins)
        cumhist *= 1.0 / len(data)

        # b is bin width
        area_above_curve = b * np.sum((1 - cumhist))
        return area_above_curve

    @staticmethod
    def run_on_single_dataframe(path_to_df_csv, label=None, output_dir=None, save=True, previous_fig_axes=None):
        """
        This method is intended to be called from an ipython notebook for plotting.

        Usage notes:
        - after calling this function, you can still change many things about the plot
        - for example you can still call plt.title("New title") to change the title
        - if you'd like to plot multiple lines on the same axes, then take the return arg of a previous call to this function, 
        - and pass it into previous_plot, i.e.:
            fig = run_on_single_dataframe("thing1.csv")
            run_on_single_dataframe("thing2.csv", previous_plot=fig)
            plt.title("both things")
            plt.show()
        - if you'd like each line to have a label in the plot, then use pass a string to label, i.e.:
            fig = run_on_single_dataframe("thing1.csv", label="thing1")
            run_on_single_dataframe("thing2.csv", label="thing2", previous_plot=fig)
            plt.title("both things")
            plt.show()

        :param path_to_df_csv: full path to csv file
        :type path_to_df_csv: string
        :param label: name that will show up labeling this line in the legend
        :type label: string
        :param save: whether or not you want to save a .png
        :type save: bool
        :param previous_plot: a previous matplotlib figure to keep building on
        :type previous_plot: None or matplotlib figure 
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        path_to_csv = utils.convert_to_absolute_path(path_to_df_csv)

        if output_dir is None:
            output_dir = os.path.dirname(path_to_csv)

        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)

        if 'is_valid_masked' not in df:
            use_masked_plots = False
        else:
            use_masked_plots = True

        if previous_fig_axes == None:
            N = 5
            if use_masked_plots:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
            else:
                fig, axes = plt.subplots(N, figsize=(10, N * 5))
        else:
            [fig, axes] = previous_fig_axes

        def get_ax(axes, index):
            if use_masked_plots:
                return axes[index, 0]
            else:
                return axes[index]

        # pixel match error
        plot = DCEP.make_pixel_match_error_plot(axes[0], df, label=label)
        axes[0].legend()
        if use_masked_plots:
            plot = DCEP.make_pixel_match_error_plot(axes[1], df, label=label, masked=True)
            axes[1].legend()
        return [fig, axes]

    @staticmethod
    def run_on_single_dataframe_across_objects(path_to_df_csv, label=None, output_dir=None, save=True,
                                               previous_fig_axes=None):
        """
        This method is intended to be called from an ipython notebook for plotting.

        See run_on_single_dataframe() for documentation.

        The only difference is that for this one, we only have across object data. 
        """
        DCEP = DenseCorrespondenceEvaluationPlotter

        path_to_csv = utils.convert_to_absolute_path(path_to_df_csv)

        if output_dir is None:
            output_dir = os.path.dirname(path_to_csv)

        df = pd.read_csv(path_to_csv, index_col=0, parse_dates=True)

        if previous_fig_axes == None:
            N = 1
            fig, ax = plt.subplots(N, figsize=(10, N * 5))
        else:
            [fig, ax] = previous_fig_axes

        # pixel match error
        plot = DCEP.make_across_object_best_match_plot(ax, df, label=label)
        ax.legend()

        if save:
            fig_file = os.path.join(output_dir, "across_objects.png")
            fig.savefig(fig_file)

        return [fig, ax]
