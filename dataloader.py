import tensorflow as tf
import os
import cv2

from tensorflow.python.ops import array_ops, math_ops


class DataLoader(object):
    """Data Loader for the SR GAN, that prepares a tf data object for training."""

    def __init__(self, input_dir, target_dir, input_size, target_size):
        """
        Initializes the dataloader.
        Args:
            image_dir: The path to the directory containing high resolution images.
            hr_image_size: Integer, the crop size of the images to train on (High
                           resolution images will be cropped to this width and height).
        Returns:
            The dataloader object.
        """
        # Sort the input image pairs
        self.input_paths = [os.path.join(input_dir, x) for x in sorted(os.listdir(input_dir))]
        self.target_paths = [os.path.join(target_dir, x) for x in sorted(os.listdir(target_dir))]
        self.target_size = target_size
        self.input_size = input_size


    def _parse_input_image(self, image_path):
        """
        Function that loads the images given the path.
        Args:
            image_path: Path to an image file.
        Returns:
            image: A tf tensor of the loaded image.
        """
        image_path = image_path.numpy().decode('utf-8')
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = tf.convert_to_tensor(image, tf.float32)

        # Check if image is large enough
        if tf.keras.backend.image_data_format() == 'channels_last':
            shape = array_ops.shape(image)[:2]
        else:
            shape = array_ops.shape(image)[1:]
        cond = math_ops.reduce_all(shape >= tf.constant(self.input_size))

        image = tf.cond(cond, lambda: tf.identity(image),
                        lambda: tf.image.resize(image, [self.input_size, self.input_size]))

        return image

    def _parse_target_image(self, image_path):
        """
        Function that loads the images given the path.
        Args:
            image_path: Path to an image file.
        Returns:
            image: A tf tensor of the loaded image.
        """

        image_path = image_path.numpy().decode('utf-8')
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = tf.convert_to_tensor(image, tf.float32)

        # Check if image is large enough
        if tf.keras.backend.image_data_format() == 'channels_last':
            shape = array_ops.shape(image)[:2]
        else:
            shape = array_ops.shape(image)[1:]
        cond = math_ops.reduce_all(shape >= tf.constant(self.target_size))

        image = tf.cond(cond, lambda: tf.identity(image),
                        lambda: tf.image.resize(image, [self.target_size, self.target_size]))

        return image



    def _random_crop(self, low_res, high_res):
        """
        Function that rescales the pixel values to the -1 to 1 range.
        For use with the generator output tanh function.
        Args:
            low_res: The tf tensor of the low res image.
            high_res: The tf tensor of the high res image. 
        Returns:
            low_res: The tf tensor of the low res image, rescaled.
            high_res: The tf tensor of the high res image, rescaled. 
        """
        Rangeoffset_h=array_ops.shape(low_res)[0]-self.input_size
        Rangeoffset_w=array_ops.shape(low_res)[1]-self.input_size
        offset_h=tf.random.uniform([1],0,Rangeoffset_h,dtype=tf.int32, seed=None)[0]
        offset_w=tf.random.uniform([1],0,Rangeoffset_w,dtype=tf.int32, seed=None)[0]
        low_res = tf.image.crop_to_bounding_box(low_res, offset_h, offset_w, self.input_size, self.input_size)
        
        offset_h_highres=int(offset_h* self.target_size/self.input_size)
        offset_w_highres=int(offset_w* self.target_size/self.input_size)
        #offset_h_highres=offset_h* self.target_size/self.input_size
        #offset_w_highres=offset_w* self.target_size/self.input_size
        high_res = tf.image.crop_to_bounding_box(high_res, offset_h_highres, offset_w_highres, self.target_size, self.target_size)
        return low_res, high_res

    def _high_low_res_pairs(self, high_res):
        # Returns low res images with the same resolution as high res 
        low_res = tf.identity(high_res)

        return low_res, high_res

    def _rescale(self, low_res, high_res):
        """
        Function that rescales the pixel values to the -1 to 1 range.
        For use with the generator output tanh function.
        Args:
            low_res: The tf tensor of the low res image.
            high_res: The tf tensor of the high res image.
        Returns:
            low_res: The tf tensor of the low res image, rescaled.
            high_res: the tf tensor of the high res image, rescaled.
        """
        high_res = high_res * 2.0 - 1.0

        return low_res, high_res

    def dataset(self, batch_size, threads=4):
        """
        Returns a tf dataset object with specified mappings.
        Args:
            batch_size: Int, The number of elements in a batch returned by the dataset.
            threads: Int, CPU threads to use for multi-threaded operation.
        Returns:
            dataset: A tf dataset object.
        """

        # Generate tf dataset from high res image paths.
        #dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        # Generate tf dataset from the input paths.
        input_dataset = tf.data.Dataset.from_tensor_slices(self.input_paths)
        # Generate tf dataset from the target path.
        target_dataset = tf.data.Dataset.from_tensor_slices(self.target_paths)

        # Read the images
        input_dataset = input_dataset.map(lambda x: tf.py_function(func = self._parse_input_image, inp = [x], Tout = tf.float32))
        target_dataset = target_dataset.map(lambda x: tf.py_function(func = self._parse_target_image, inp = [x], Tout = tf.float32))

        # Generate low resolution by downsampling crop.
        #dataset = dataset.map(self._high_low_res_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

        # Crop out a piece for training
        dataset = dataset.map(self._random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #input_dataset = input_dataset.map(self._input_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #target_dataset = target_dataset.map(self._target_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Rescale the values in the input
        dataset = dataset.map(self._rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batch the input, drop remainder to get a defined batch size.
        # Prefetch the data for optimal GPU utilization.
        dataset = dataset.shuffle(30).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
