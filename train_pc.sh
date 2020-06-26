#!/bin/bash

source ../bin/activate
python --version
python main.py --input_dir ../data/train_data/training_10_images/10_HDR_EXR_PQ --input_size 96 --target_dir ../data/train_data/training_10_images/10_SDR_EΧΡ_Gamma --target_size 96 --batch_size 5 --epochs 1 --lr 1e-4 --save_iter 2 --dis ../models/discriminator.h5 --gen ../models/generator.h5


