import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet_utils
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet_mod(models.ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet_mod, self).__init__(*args,**kwargs)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x,1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_mod(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', resnet_utils.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', resnet_utils.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', resnet_utils.Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `stride` + 1.
    This allows the proper alignment of features
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / stride).astype(np.int) * stride + 1

    # Converting the numpy to list, torch.nn.functional.interpolate accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.interpolate(input=input_img_batch,
                                                         size=new_spatial_dims)

    return input_img_batch_new_size


class Resnet34_8s(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(pretrained=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        
        self.resnet34_8s = resnet34_8s
        
        self._normal_initialization(self.resnet34_8s.fc)
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, stride=8)
        
        x = self.resnet34_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x


class Resnet50_8s(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(pretrained=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, stride=8)
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x


class Resnet101_8s(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet101_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = resnet101(pretrained=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_8s.fc = nn.Conv2d(resnet101_8s.inplanes, num_classes, 1)
        
        self.resnet101_8s = resnet101_8s
        
        self._normal_initialization(self.resnet101_8s.fc)
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, stride=8)
        
        x = self.resnet101_8s(x)
        
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear')
        
        return x
