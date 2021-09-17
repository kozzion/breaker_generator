import sys

import numpy as np
import cv2
import time

sys.path.append('../')
from breaker_generator.video.videosource_webcam_video_capture import VideosourceWebcamVideoCapture

video_source = VideosourceWebcamVideoCapture(target_width=256, target_height=256)
try:
    video_source.start()
    for i in range(100):
        print('frame: ' + str(i))
        sys.stdout.flush()

        frame = video_source.queue_output.get()
        cv2.imshow("video", frame)
        key = cv2.waitKey(1)


            # show one frame at a time
        # cv2.waitKey(00) == ord('k')
        #     # Quit when 'q' is pressed
        # if cv2.waitKey(1) == ord('q'):
        #     break
    video_source.stop()
except:
    video_source.stop()
