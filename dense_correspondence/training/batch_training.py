# +
import os, sys
sys.path.append(os.getcwd() + "/../../modules")
sys.path.append(os.getcwd() + "/../../external")
sys.path.append(os.getcwd() + "/../..")
sys.path.append(os.getcwd() + "/../dataset")

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

# utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([0, 2]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
logging.basicConfig(level=logging.INFO)


# +
dc_source_dir = utils.getDenseCorrespondenceSourceDir()

dataset_name = 'shirt_all-poses_white'
config_filename = os.path.join(dc_source_dir, 'config/dense_correspondence/dataset/composite/%s.yaml' % dataset_name)
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(dc_source_dir, 'config/dense_correspondence/training/training.yaml')
train_config = utils.getDictFromYamlFilename(train_config_file)

dataset = SpartanDataset(config=config)

logging_dir = os.path.join(dc_source_dir, "pdc/trained_models/tutorials")

# fixed params
loss_type = 'distributional'
normalize = False
lr = 1.0e-4
sigma = 1
net_type = 'resnet101'

# tunable params
num_iterations = 4000
d = 9

name = "%s_d%d_resnet101" % (dataset_name, d)
print("\nModel name:", name)

train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["training"]["num_iterations"] = num_iterations
train_config["training"]["learning_rate"] = lr
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["dense_correspondence_network"]["normalize"] = normalize
train_config["dense_correspondence_network"]["loss_type"] = loss_type
train_config["dense_correspondence_network"]["sigma"] = sigma
train_config["dense_correspondence_network"]["net_type"] = net_type

TRAIN = True
EVALUATE = True
# -

if TRAIN:
    start = time.time()
    print("training descriptor of dimension %d" %(d))
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
    print("finished training descriptor of dimension %d" %(d))
    print("training took %d seconds" %(time.time() - start))

if EVALUATE:
    model_folder = os.path.join(logging_dir, name)
    model_folder = utils.convert_to_absolute_path(model_folder)
    DCE = DenseCorrespondenceEvaluation
    num_image_pairs = 100
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs) 
