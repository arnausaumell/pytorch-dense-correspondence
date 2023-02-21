import torch
import torch.nn.functional as F
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

class PixelwiseDistributionalLoss(object):

    def __init__(self, image_shape, config=None):
        self.type = "pixelwise_distributional"
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]

        self._config = config

        self._debug_data = dict()

        self._debug = False

    @property
    def debug(self):
        return self._debug

    @property
    def config(self):
        return self._config

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def debug_data(self):
        return self._debug_data
    
    
    def flattened_mask_indices(self, img_mask, inverse=True):
        try:
            height, width = img_mask.shape[1:]
        except:
            height, width = img_mask.shape
        mask = img_mask.view(width*height, 1).squeeze(1)
        if inverse:
            inv_mask = 1 - mask
            inv_mask_indices_flat = torch.nonzero(inv_mask)
            return inv_mask_indices_flat
        else:
            return torch.nonzero(mask)

        
    def gauss_2d_distribution(self, width, height, sigma, u, v, masked_indices=None, symmetry=True):
        X,Y=np.meshgrid(np.linspace(0,width,width),np.linspace(0,height,height))
        mu_x = u
        mu_y = v
        if symmetry:
            mu_x2 = width-1-u
            G1=np.exp(-((X-mu_x)**2+(Y-mu_y)**2)/(2.0*sigma**2)).ravel()
            G2=np.exp(-((X-mu_x2)**2+(Y-mu_y)**2)/(2.0*sigma**2)).ravel()
            G = 0.5*G1 + 0.5*G2
        else:
            G=np.exp(-((X-mu_x)**2+(Y-mu_y)**2)/(2.0*sigma**2)).ravel()
        
        if masked_indices is not None:
            G[masked_indices] = 0.0
        G /= G.sum()
        G += 1e-300
        return torch.from_numpy(G).double().cuda()

    
    def distributional_loss_single_match(self, i, img_a, img_b, image_a_pred, image_b_pred, match_a, match_b, 
                                         masked_indices=None, sigma=1, symmetry=True):
        image_height, image_width = img_a.shape[2:]
        match_b_descriptor = torch.index_select(image_b_pred, 1, match_b) # get descriptor for image_b at match_b
        norm_degree = 2
        descriptor_diffs = image_a_pred.squeeze() - match_b_descriptor.squeeze()
        norm_diffs = descriptor_diffs.norm(norm_degree, 1).pow(2)
        p_a = F.softmax(-1 * norm_diffs, dim=0).double() # compute current distribution         
        u,v = (match_a.item()%image_width, match_a.item()//image_width)
        q_a = self.gauss_2d_distribution(image_width, image_height, sigma, u, v, 
                                         masked_indices=masked_indices, symmetry=symmetry)
        loss = F.kl_div(p_a.log(), q_a, reduction='sum', log_target=False) # compute kl divergence loss 
#         loss = F.kl_div(q_a.log(), p_a, reduction='sum', log_target=False) # compute kl divergence loss 
#         if i%300 == 0 and i==0:
#             _, ax = plt.subplots(1,3,figsize=(15,10))
#             ax[0].imshow(q_a.cpu().detach().numpy().reshape(image_height, image_width))
#             ax[1].imshow(p_a.cpu().detach().numpy().reshape(image_height, image_width))
#             ax[2].imshow(img_a.squeeze().transpose(0,2).transpose(0,1).cpu().detach().numpy())
#             plt.show()
        return loss

    def get_loss(self, img_a, img_b, image_a_pred, image_b_pred, matches_a, matches_b, image_a_mask, image_b_mask, sigma=1, symmetry=True):
        loss = 0.0
        masked_indices_a = self.flattened_mask_indices(image_a_mask, inverse=True)
        masked_indices_b = self.flattened_mask_indices(image_b_mask, inverse=True)
        matches_lists = list(zip(matches_a, matches_b))
        matches_lists = random.sample(matches_lists, min(50, len(matches_lists)))
        for i, (match_a, match_b) in enumerate(matches_lists):
            loss += self.distributional_loss_single_match(i, img_b, img_a, image_b_pred, image_a_pred, match_b, match_a, 
                                                          masked_indices=masked_indices_b, sigma=sigma, symmetry=symmetry)
        return loss/len(matches_lists)
