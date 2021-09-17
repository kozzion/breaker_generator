import numpy as np
import pygame
import pygame.camera
import threading
import time
from queue import Queue

from breaker_generator.image.array_image_to_array_image import ArrayImageToArrayImage

class VideosourceWebcamPygame:





        # cam = pygame.camera.Camera("/dev/video0", (640, 480))

        # time.sleep(0.1)  # You might need something higher in the beginning
        # img = cam.get_image()
        # pygame.image.save(img, "pygame.jpg")
        # cam.stop()
    # also requires videocapture http://videocapture.sourceforge.net/ from https://www.lfd.uci.edu/~gohlke/pythonlibs/#videocapture
    def __init__(self, target_width, target_height, delay_frame_s=0.014, max_queue_size=1) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.converter = ArrayImageToArrayImage(
            list_size_output    = [self.target_width, self.target_height], 
            format_array_input  = 'rgb_0255_uint8_cl', 
            format_array_output = 'rgb_0255_uint8_cl', 
            format_crop         = 'cc_rs')

        self.delay_frame_s = delay_frame_s
        self.max_queue_size = max_queue_size
        self.frame_last = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        pygame.camera.init()
        self.camera = None
        self.thread_run = threading.Thread(target=self.run)
        self.is_running = False
        self.queue_output = Queue(max_queue_size)

    def get_list_cameras(self):
        return pygame.camera.list_cameras()[0]


    def start(self):
        self.thread_run.start()

    def run(self):
        self.is_running = True
        # self.camera = pygame.camera.Camera(self.name_device, (640, 480))
        self.camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
        self.camera.start()
        while self.is_running:
            frame = self.camera.get_image()
            frame_converted, metadata = self.converter.convert(frame, {})
            self.frame_last = frame_converted
            self.queue_output.put(frame_converted)
            self.queue_output.put(self.frame_last)
            time.sleep(self.delay_frame_s)
         

    def stop(self):
        self.is_running = False
        self.thread_run.join()
        self.camera.stop()
