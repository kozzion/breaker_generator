import pyvirtualcam
import numpy as np

class VirualWebcam:

    # install unity capture for virual camera
    # https://github.com/schellingb/UnityCapture
    
    def __init__(self, width=1280, height=720) -> None:
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)  # RGB

    def set_frame(self, frame):
        if frame.shape != self.frame.shape:
            raise Exception('shapes do not match')
        self.frame = frame


    def start(self):
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=20) as cam:
            print(f'Using virtual camera: {cam.device}')
            while True:
                # self.frame[:] = cam.frames_sent % 255  # grayscale animation
                cam.send(self.frame)
                cam.sleep_until_next_frame()
   
