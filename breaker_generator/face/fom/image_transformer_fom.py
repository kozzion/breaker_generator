import sys
import os
import json

import numpy as np
import torch

from breaker_generator.face.fom.tools_fom import ToolsFom
from breaker_generator.face.fom.keypoint_detector import KPDetector
from breaker_generator.face.fom.generator import OcclusionAwareGenerator
from breaker_generator.face.fom.sync_batchnorm import DataParallelWithCallback

# based on https://aliaksandrsiarohin.github.io/first-order-model-website/
class ImageTranformerFom:

    def __init__(self, path_file_config, path_file_checkpoint, array_image_source, mode_relative=True, mode_adapt_scale=True, mode_compute_gpu=True) -> None:
        self.path_file_config = path_file_config
        self.path_file_checkpoint = path_file_checkpoint
        self.mode_relative=mode_relative 
        self.mode_adapt_scale=mode_adapt_scale 
        self.mode_compute_gpu=mode_compute_gpu


        self.kp_source = None
        self.generator = None
        self.kp_detector = None
        self.load_checkpoints()

        self.array_image_source_torch = torch.tensor(array_image_source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) 
        if self.mode_compute_gpu == False:
            self.array_image_source_torch = self.array_image_source_torch.cuda()

    
    def load_checkpoints(self):

        with open(self.path_file_config) as file:
            config = json.load(file)

        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    

        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if self.mode_compute_gpu:
            self.kp_detector.cuda()
            self.generator.cuda()
            checkpoint = torch.load(self.path_file_checkpoint)                
            self.generator.load_state_dict(checkpoint['generator'])
            self.kp_detector.load_state_dict(checkpoint['kp_detector'])
            self.generator = DataParallelWithCallback(self.generator)
            self.kp_detector = DataParallelWithCallback(self.kp_detector)

        else:
            checkpoint = torch.load(self.path_file_checkpoint, map_location=torch.device('cpu'))
            self.generator.load_state_dict(checkpoint['generator'])
            self.kp_detector.load_state_dict(checkpoint['kp_detector'])
        
        self.generator.eval()
        self.kp_detector.eval()
        
    def restart(self):
        self.kp_source = None

    def set_array_image_source(self, array_image_source):
        self.array_image_source_torch = torch.tensor(array_image_source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) 
        if self.mode_compute_gpu == False:
            self.array_image_source_torch = self.array_image_source_torch.cuda()

    def transform(self, array_image_driving):
        #256 256 01 f32 cl(?)

        array_image_driving_torch = torch.tensor(np.array(array_image_driving)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if self.mode_compute_gpu:
            array_image_driving_torch = array_image_driving_torch.cuda()

        if self.kp_source == None: 
            self.kp_source = self.kp_detector(self.array_image_source_torch)
            self.kp_driving_initial = self.kp_detector(array_image_driving_torch)

        kp_driving = self.kp_detector(array_image_driving_torch)

        kp_norm = ToolsFom.normalize_kp(
            kp_source=self.kp_source, 
            kp_driving=kp_driving,
            kp_driving_initial=self.kp_driving_initial, 
            use_relative_movement=self.mode_relative,
            use_relative_jacobian=self.mode_relative, 
            adapt_movement_scale=self.mode_adapt_scale)

        out = self.generator(
            self.array_image_source_torch, 
            kp_source=self.kp_source, 
            kp_driving=kp_norm)

        return np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

  
