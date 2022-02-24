import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
import time
from PIL import Image
import numbers

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod, QuantizationMode


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, size_limit=0, augmented_model_path='resnet18_fp32.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.sizelimit = size_limit
        self.datasize, self.enum_data_dicts = self._preproc()

    def _preproc(self):
        session = onnxruntime.InferenceSession(self.augmented_model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        nhwc_data_list = preprocess_func(
            self.image_folder, height, width, self.sizelimit)
        input_name = session.get_inputs()[0].name
        return len(nhwc_data_list), iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])

    def get_next(self):
        return next(self.enum_data_dicts, None)


"""
def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - \
        np.array([123.68, 116.78, 103.94], dtype=np.float32)
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data
"""


def img_resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)


def img_crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))


def img_center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)


def preprocess(input_data):

    img_data = input_data.astype('float32')

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])

    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')

    return norm_img_data


def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    image_names.sort()
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        print("image_filepath=", image_filepath)
        image = Image.open(image_filepath).convert("RGB")
        image = img_resize(image, 256)
        image = img_center_crop(image, 224)
        image_data = np.array(image).transpose(2, 0, 1)
        input_data = preprocess(image_data)
        unconcatenated_batch_data.append(input_data)
    batch_data = np.concatenate(np.expand_dims(
        unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_model", required=True, help="input model")
    # parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--input_model", default="resnet18_fp32.onnx")
    parser.add_argument("--output_model", default="resnet50.quant.onnx")
    parser.add_argument("--calibrate_dataset",
                        default="./test_images", help="calibration data set")
    parser.add_argument("--quant_format",
                        default=QuantFormat.PTQEnflame,
                        type=QuantFormat.from_string,
                        choices=list(QuantFormat))
    parser.add_argument("--per_channel", default=True, type=bool)
    parser.add_argument("--size_limit",
                        # default=0,
                        default=980,
                        type=int
                        )
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = ResNet50DataReader(calibration_dataset_path, args.size_limit)

    quantize_static(input_model_path,
                    output_model_path,
                    dr,
                    quant_format=QuantFormat.PTQEnflame,
                    per_channel=args.per_channel,
                    nodes_to_exclude=[],
                    activation_type=QuantType.QInt8,
                    nodes_to_maxthreshold=[],
                    weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.EntropyEnflame,
                    quantization_mode=QuantizationMode.QEnflameOps)
    print('Calibrated and quantized model saved.')

    print('benchmarking int8 model...')


if __name__ == '__main__':
    main()
