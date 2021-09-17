import numpy as np
import vidcap

import threading
import time
from queue import Queue
from VideoCapture import Device

from breaker_generator.image.array_image_to_array_image import ArrayImageToArrayImage





cam0 = Device(0)
# cam1 = Device(1)

# for i in xrange(30):
#     cam0.saveSnapshot('video/image0_%d.jpg' % i, timestamp=1)
#     time.sleep(0.05)
#     cam1.saveSnapshot('video/image1_%d.jpg' % i, timestamp=1)
#     time.sleep(0.1)
class VideosourceWebcamVideoCapture:





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

        self.camera = None
        self.name_device = None
        self.thread_run = threading.Thread(target=self.run)
        self.is_running = False
        self.queue_output = Queue(max_queue_size)




    def start(self):
        self.thread_run.start()


    def run(self):
        try:
            self.is_running = True
            # self.camera = pygame.camera.Camera(self.name_device, (640, 480))
            print('here3')
            self.camera = Device(0)
            print('here4')
            while self.is_running:
                print('here0')
                self.camera.saveSnapshot('image0.jpg', timestamp=1)
                print('here1')
                time.sleep(0.05)
                frame = self.camera.getImage(self, timestamp=0, boldfont=0, textpos='bl')
                print('here2')
                frame_converted, metadata = self.converter.convert(frame, {})
                self.frame_last = frame_converted
                self.queue_output.put(frame_converted)
                time.sleep(self.delay_frame_s)
        except:
            self.is_running = False

    def get_array_image(self):
        if self.is_running:
            return self.queue_output.get()


    def stop(self):
        self.is_running = False
        self.thread_run.join()
