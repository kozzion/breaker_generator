import sys

import numpy as np
import cv2
import time

sys.path.append('../')
from breaker_generator.image.array_image_to_array_image import ArrayImageToArrayImage

array_image_source = np.zeros((512, 512, 3), dtype=np.uint8)

converter = ArrayImageToArrayImage(      
            list_size_output    = [256, 256], 
            format_array_input  = 'rgb_0255_uint8_cl', 
            format_array_output = 'rgb_0255_uint8_cl', 
            format_crop         = 'cc_rs')
array_image_target, metatdate = converter.convert(array_image_source, {})
print(array_image_target.shape)
print(array_image_target.dtype)


converter = ArrayImageToArrayImage(      
            list_size_output    = [256, 256], 
            format_array_input  = 'rgb_0255_uint8_cl', 
            format_array_output = 'rgb_0255_f32_cl', 
            format_crop         = 'cc_rs',
            do_batch=True)
array_image_target, metatdate = converter.convert(array_image_source, {})
print(array_image_target.shape)
print(array_image_target.dtype)