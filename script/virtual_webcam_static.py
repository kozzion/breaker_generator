import sys
import os
import json
import numpy as np
from PIL import Image
import time

from skimage.transform import resize

sys.path.append('../')
from breaker_generator.video.virtual_webcam import VirualWebcam

with open('config.cfg', 'r') as file:
    config = json.load(file)

path_dir_instancegroup = 'C:\\project\\data\\data_breaker\\instancegroup\\ig-test'
path_file_image_source = os.path.join(path_dir_instancegroup, 'test.jpg')
array_image_source = np.asarray(Image.open(path_file_image_source))
# array_image_source = (resize(array_image_source, (720, 1280))[..., :3]).astype(np.uint8)

array_image_source = array_image_source[:256,:256,:]
virtual_webcam = VirualWebcam()
virtual_webcam.set_frame(array_image_source)
virtual_webcam.start()
print('press any key')
input()
virtual_webcam.stop()
#generator.generate()