import os
import sys

import cv2
import numpy as np
import imageio
from skimage.transform import resize

sys.path.append('../')
from breaker_generator.video.videosource_webcam_opencv import VideosourceWebcamOpencv
from breaker_generator.image.array_image_to_array_image import ArrayImageToArrayImage
from breaker_generator.face.fom.image_transformer_fom import ImageTranformerFom

converter_pre = ArrayImageToArrayImage(
                list_size_output    = [256, 256], 
                format_array_input  = 'rgb_0255_uint8_cl', 
                format_array_output = 'rgb_01_f32_cl', 
                format_crop         = 'cc_rs')  


converter_post = ArrayImageToArrayImage(
                list_size_output    = [256, 256], 
                format_array_input  = 'rgb_01_f32_cl', 
                format_array_output = 'rgb_0255_uint8_cl', 
                format_crop         = 'cc_rs')  

video_source = VideosourceWebcamOpencv(target_width=256, target_height=256)


path_dir_encoder = 'C:\\project\\data\\data_breaker\\encoder\\en-fomvox'
path_file_config = os.path.join(path_dir_encoder, 'config_torch.json')
path_file_checkpoint = os.path.join(path_dir_encoder, 'checkpoint.pth.tar')

# path_file_image_source = '00.png'
path_dir_instancegroup = 'C:\\project\\data\\data_breaker\\instancegroup\\ig-test'
path_file_image_source = os.path.join(path_dir_instancegroup, '00.png')
path_file_video_driving = os.path.join(path_dir_instancegroup, '00.mp4')
path_file_result = os.path.join(path_dir_instancegroup, 'result.mp4')

array_image_source = imageio.imread(path_file_image_source)
array_image_source = resize(array_image_source, (256, 256))[..., :3]

transformer = ImageTranformerFom(path_file_config, path_file_checkpoint, array_image_source)


video_source.start()
index = 0
while True:
    print('frame: ' + str(index))
    sys.stdout.flush()
    index += 1

    frame = video_source.queue_output.get()
    cv2.imshow("pre", frame)
    array_image_pre, metadata = converter_pre.convert(frame, {})
    array_image_trans = transformer.transform(array_image_pre)
    array_image_post, metadata = converter_post.convert(array_image_trans, {})
    cv2.imshow("post", array_image_post)
    key = cv2.waitKey(14)
    # print(key)
    sys.stdout.flush()
    if key != -1:
        print(key)
        sys.stdout.flush()
        if key == 27:
            print('break')
            break
        if key == 114:
            print('restart')
            transformer.restart()

        # show one frame at a time
    # cv2.waitKey(00) == ord('k')
    #     # Quit when 'q' is pressed
    # if cv2.waitKey(1) == ord('q'):
    #     break
video_source.stop()
