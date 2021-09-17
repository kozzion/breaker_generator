import numpy as np
import threading
import cv2
import time
from queue import Queue
import sys

from breaker_generator.image.array_image_to_array_image import ArrayImageToArrayImage

class VideosourceWebcamOpencv:

    def __init__(self, target_width, target_height, delay_s=0.014, converter=None, max_queue_size=1) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.delay_s = delay_s
        if converter:
            self.converter = converter 
        else:
            self.converter = ArrayImageToArrayImage(
                list_size_output    = [target_height, target_width], 
                format_array_input  = 'rgb_0255_uint8_cl', 
                format_array_output = 'rgb_0255_uint8_cl', 
                format_crop         = 'cc_rs')       

        self.frame_last = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
    
        self.thread_run = threading.Thread(target=self.run)
        self.is_running = False
        self.queue_output = Queue(max_queue_size)

    def set_frame(self, frame):
        if frame.shape != self.frame.shape:
            raise Exception('shapes do not match')
        self.frame = frame


    def start(self):
        self.thread_run.start()

    def run(self, fail_hard=True):
        try:
            self.is_running = True
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                raise IOError("Cannot open webcam")
                
            while self.is_running:
                ret, frame = video_capture.read()
 
                while (ret == False):
                    time.sleep(0.1)
                    ret, frame = video_capture.read()
                frame_converted, metadata = self.converter.convert(frame, {})
                
                self.frame_last = frame_converted
                self.queue_output.put(frame_converted)

                time.sleep(self.delay_s)
        except Exception as e:
            if fail_hard:
                exit()
            else:
                raise e

    def stop(self):
        self.is_running = False
        while (0 < self.queue_output.qsize()):
            self.queue_output.get_nowait()
        self.thread_run.join()

    def get_frame(self):
        return self.frame_last
