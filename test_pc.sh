#!/bin/bash
source ../bin/activate
python --version
python infer_EXR.py --image_dir ../data/train_data/training_10_images/10_HDR_EXR_PQ --output_dir /Users/yuyang/571_sub/data/test_data/results


