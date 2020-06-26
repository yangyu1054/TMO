from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import cv2
import os
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')


def main():

    # limits tensorflow memory demand
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)



    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # Change model input shape to accept all size inputs
    model = keras.models.load_model('../models/generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    # Loop over all images
    for image_path in image_paths:
        
        # Read image
        low_res= cv2.imread(image_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  

        # Convert to RGB (opencv uses BGR as default but the images are usually in RGB format - so we change the order of colors)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        #  Rescale values to 0-1  (the model has output between [-1,1] so we add 1 to the image and divide by two)
        sr = ((sr + 1) / 2.);

        # Convert back to BGR for opencv  (BGR->RGB)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        # Save the results: write EXR
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)


if __name__ == '__main__':
    main()
