import sys
import os
import json
import numpy as np
import cv2
import time
import imageio

sys.path.append('../')
from breaker_generator.video.virtual_webcam import VirualWebcam
from breaker_generator.face.fom.image_transformer_fom import ImageTranformerFom

with open('config.cfg', 'r') as file:
    config = json.load(file)

def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            print('camera.get(0)')
            print(camera.get(0))
            print(camera.get(1))
            print(camera.get(2))
            print(camera.get(3))
            print(camera.get(4))
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports, working_ports


available_ports,working_ports =  list_ports()
print(available_ports)
print(working_ports)
## webcam
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

## AI
path_dir_encoder = 'C:\\project\\data\\data_breaker\\encoder\\en-fomvox'
path_file_config = os.path.join(path_dir_encoder, 'config_torch.json')
path_file_checkpoint = os.path.join(path_dir_encoder, 'checkpoint.pth.tar')

## indentity
path_dir_instancegroup = 'C:\\project\\data\\data_breaker\\instancegroup\\ig-test'
path_file_image_source = os.path.join(path_dir_instancegroup, '00.png')
array_image_source = imageio.imread(path_file_image_source)
transformer = ImageTranformerFom(path_file_config, path_file_checkpoint, array_image_source)

ret, frame = vc.read()
while not ret:
    ret, frame = vc.read()
offset_0 = 100
offset_1 = 200
array_image_source = frame[offset_0:offset_0 + 256, offset_1: offset_1 + 256, :]


# virtual_webcam = VirualWebcam()
# virtual_webcam.set_frame(array_image_source)
# virtual_webcam.start()

for array_image_source in range(200):
    ret, frame = vc.read()
    array_image_driving = frame[offset_0:offset_0 + 256, offset_1: offset_1 + 256, :]
    array_image_target = transformer.transform(array_image_driving)
    # virtual_webcam.set_frame(array_image_source)
    time.sleep(0.01)
# virtual_webcam.stop()
#generator.generate()