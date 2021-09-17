import matplotlib
matplotlib.use('Agg')
import os
import sys

import imageio

from skimage.transform import resize
from skimage import img_as_ubyte



sys.path.append('../')
from breaker_generator.face.fom.image_transformer_fom import ImageTranformerFom

path_dir_encoder = 'C:\\project\\data\\data_breaker\\encoder\\en-fomvox'
path_file_config = os.path.join(path_dir_encoder, 'config_torch.json')
path_file_checkpoint = os.path.join(path_dir_encoder, 'checkpoint.pth.tar')

# path_file_image_source = '00.png'
path_dir_instancegroup = 'C:\\project\\data\\data_breaker\\instancegroup\\ig-test'
path_file_image_source = os.path.join(path_dir_instancegroup, '00.png')
path_file_video_driving = os.path.join(path_dir_instancegroup, '00.mp4')
path_file_result = os.path.join(path_dir_instancegroup, 'result.mp4')

array_image_source = imageio.imread(path_file_image_source)
reader = imageio.get_reader(path_file_video_driving)
fps = reader.get_meta_data()['fps']

driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

array_image_source = resize(array_image_source, (256, 256))[..., :3]
list_frame_driving = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

transformer = ImageTranformerFom(path_file_config, path_file_checkpoint, array_image_source)

list_frame_target = []
import torch
with torch.no_grad():
    for array_image_driving in list_frame_driving:
        array_image_target = transformer.transform(array_image_driving)
        list_frame_target.append(array_image_target)


imageio.mimsave(path_file_result, [img_as_ubyte(frame) for frame in list_frame_target], fps=fps)