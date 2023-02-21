import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType


# +
def get_contrastive_loss(pixelwise_contrastive_loss, match_type,
                         img_a, img_b,
                         image_a_pred, image_b_pred,
                         matches_a, matches_b,
                         masked_non_matches_a, masked_non_matches_b,
                         background_non_matches_a, background_non_matches_b,
                         blind_non_matches_a, blind_non_matches_b, symmetry):
    """
    This function serves the purpose of:
    - parsing the different types of SpartanDatasetDataType...
    - parsing different types of matches / non matches..
    - into different pixelwise contrastive loss functions

    :return args: loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss
    :rtypes: each pytorch Variables

    """
    pcl = pixelwise_contrastive_loss
    
    image_height, image_width = img_a.shape[2:]
    
#     plt.imshow(img_a.cpu())
#     plt.imshow(img_b.cpu())
#     plt.show()

    match_loss, masked_non_match_loss, num_masked_hard_negatives = \
        pixelwise_contrastive_loss.get_loss_matched_and_non_matched_with_l2(image_a_pred, image_b_pred,
                                                                            image_height, image_width,
                                                                            matches_a, matches_b,
                                                                            masked_non_matches_a, masked_non_matches_b,
                                                                            symmetry, M_descriptor=pcl._config["M_masked"])

    if pcl._config["use_l2_pixel_loss_on_background_non_matches"]:
        background_non_match_loss, num_background_hard_negatives = \
            pixelwise_contrastive_loss.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b,
                                                                         background_non_matches_a,
                                                                         background_non_matches_b,
                                                                         M_descriptor=pcl._config["M_background"])

    else:
        background_non_match_loss, num_background_hard_negatives = \
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                      background_non_matches_a,
                                                                      background_non_matches_b,
                                                                      M_descriptor=pcl._config["M_background"])

    blind_non_match_loss = zero_loss()
    num_blind_hard_negatives = 1
    if not (SpartanDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_blind_hard_negatives = \
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                      blind_non_matches_a, blind_non_matches_b,
                                                                      M_descriptor=pcl._config["M_masked"])

    total_num_hard_negatives = num_masked_hard_negatives + num_background_hard_negatives
    total_num_hard_negatives = max(total_num_hard_negatives, 1)

    if pcl._config["scale_by_hard_negatives"]:
        scale_factor = total_num_hard_negatives

        masked_non_match_loss_scaled = masked_non_match_loss * 1.0 / max(num_masked_hard_negatives, 1)

        background_non_match_loss_scaled = background_non_match_loss * 1.0 / max(num_background_hard_negatives, 1)

        blind_non_match_loss_scaled = blind_non_match_loss * 1.0 / max(num_blind_hard_negatives, 1)
    else:
        # we are not currently using blind non-matches
        num_masked_non_matches = max(len(masked_non_matches_a), 1)
        num_background_non_matches = max(len(background_non_matches_a), 1)
        num_blind_non_matches = max(len(blind_non_matches_a), 1)
        scale_factor = num_masked_non_matches + num_background_non_matches

        masked_non_match_loss_scaled = masked_non_match_loss * 1.0 / num_masked_non_matches

        background_non_match_loss_scaled = background_non_match_loss * 1.0 / num_background_non_matches

        blind_non_match_loss_scaled = blind_non_match_loss * 1.0 / num_blind_non_matches

    non_match_loss = 1.0 / scale_factor * (masked_non_match_loss + background_non_match_loss)

    loss = pcl._config["match_loss_weight"] * match_loss + \
           pcl._config["non_match_loss_weight"] * non_match_loss

    return loss, match_loss, masked_non_match_loss_scaled, background_non_match_loss_scaled, blind_non_match_loss_scaled


# -

def get_distributional_loss(pixelwise_distributional_loss,
                            img_a, img_b,
                            image_a_pred, image_b_pred,
                            matches_a, matches_b,
                            image_a_mask, image_b_mask, 
                            sigma, symmetry):
    """
    This function serves the purpose of:
    - parsing the different types of SpartanDatasetDataType...
    - parsing different types of matches / non matches..
    - into different pixelwise contrastive loss functions

    :return args: loss, match_loss, masked_non_match_loss, \
                background_non_match_loss, blind_non_match_loss
    :rtypes: each pytorch Variables

    """
    
    return pixelwise_distributional_loss.get_loss(img_a, img_b,
                                                  image_a_pred, image_b_pred, 
                                                  matches_a, matches_b, 
                                                  image_a_mask, image_b_mask,
                                                  sigma=sigma, symmetry=symmetry)


def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())


def is_zero_loss(loss):
    return loss.data[0] < 1e-20
