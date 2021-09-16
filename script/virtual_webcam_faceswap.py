import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.append('../')
from breaker_generator.video.virtual_webcam import VirualWebcam

with open('config.cfg', 'r') as file:
    config = json.load(file)

path_file_image = 'test.jpg'
frame = np.asarray(Image.open(path_file_image))

print(frame.shape)

virtual_webcam = VirualWebcam()
virtual_webcam.set_frame(frame)
virtual_webcam.start()
#generator.generate()