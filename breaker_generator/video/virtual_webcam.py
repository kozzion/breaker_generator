import sys
import pyvirtualcam
import numpy as np
import time
import threading

class VirualWebcam:

    # install unity capture for virual camera
    # https://github.com/schellingb/UnityCapture
    
    # def __init__(self, width=1280, height=720) -> None:
    def __init__(self, width=256, height=256) -> None:
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)  # RGB
        self.thread_run = threading.Thread(target=self.run)
        self.is_running = False

    def set_frame(self, frame):
        if frame.shape != self.frame.shape:
            raise Exception('shapes do not match')
        self.frame = frame


    def start(self):
        self.thread_run.start()

    def run(self):
        self.is_running = True
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=30) as cam:
            while self.is_running:
                # print('loop')
                # sys.stdout.flush()
                # self.frame[:] = cam.frames_sent % 255  # grayscale animation
                cam.send(self.frame)
                #cam.sleep_until_next_frame()

    def stop(self):
        self.is_running = False
        self.thread_run.join()

   
