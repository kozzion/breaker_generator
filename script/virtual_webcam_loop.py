import sys
import os
import json
import numpy as np
from PIL import Image
import time
import imageio

from skimage.transform import resize

sys.path.append('../')
from breaker_generator.video.virtual_webcam import VirualWebcam

with open('config.cfg', 'r') as file:
    config = json.load(file)

path_dir_instancegroup = 'C:\\project\\data\\data_breaker\\instancegroup\\ig-test'
path_file_video_source = os.path.join(path_dir_instancegroup, '00.mp4')
reader = imageio.get_reader(path_file_video_source)
fps = reader.get_meta_data()['fps']

list_array_image_source = []
try:
    for frame in reader:
        list_array_image_source.append(frame)
except RuntimeError:
    pass
reader.close()

virtual_webcam = VirualWebcam()
virtual_webcam.set_frame(list_array_image_source[0])
virtual_webcam.start()

for array_image_source in list_array_image_source:
    virtual_webcam.set_frame(array_image_source)
    time.sleep(0.01)
virtual_webcam.stop()
#generator.generate()