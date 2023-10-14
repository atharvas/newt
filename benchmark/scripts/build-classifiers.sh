#!/bin/bash -i
# This file builds a classifier model containing all the classifiers + pretrained resnet.
# Make sure this file is executed from newt/benchmark directory.
conda activate newtenv
set -x
{
CUDA_VISIBLE_DEVICES=1 python build_classifiers.py --best_performing_model_tag inat2021_supervised --best_performing_model_loc /home/asehgal/viper/data/inat/cvpr21_newt_pretrained_models/pt/inat2021_supervised_large.pth.tar --save_dir /home/asehgal/viper/pretrained_models/newt/
} &>> logs/build-classifiers.log