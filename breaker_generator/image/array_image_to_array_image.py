import sys
import os
import imageio
import numpy as np

from breaker_generator.image.tools_array_image import ToolsArrayImage 

class ArrayImageToArrayImage(object):                
    def __init__(self, *, list_size_output=None, format_array_input='rgb_0255_uint8_cl', format_array_output = 'rgb_0255_f32_cl', format_crop = 'cp_rs', do_batch=False):
        super(ArrayImageToArrayImage, self).__init__()
        self.list_size_output = list_size_output
        self.do_batch = do_batch
        self.format_array_input = format_array_input
        self.format_array_output = format_array_output
        self.format_crop = format_crop



        self._input_color_mode = self.format_array_input.split('_')[0]
        self._input_value_range = self.format_array_input.split('_')[1]
        self._input_data_type = self.format_array_input.split('_')[2]
        self._input_channel_place = self.format_array_input.split('_')[3]  


        self._output_color_mode = self.format_array_output.split('_')[0]
        self._output_value_range = self.format_array_output.split('_')[1]
        self._output_data_type = self.format_array_output.split('_')[2]
        self._output_channel_place = self.format_array_output.split('_')[3]  



        if not self._input_color_mode in ['bgr', 'bgra', 'rgb', 'rgba']:
            raise NotImplementedError()

        if not self._input_value_range in ['01', '0255']:
            raise NotImplementedError()

        if not self._input_data_type in ['uint8', 'f32']:
            raise NotImplementedError()

        if not self._input_channel_place in ['cl']:
            raise NotImplementedError()



        if not self._output_color_mode in ['rgb', 'bgr']:
            raise NotImplementedError()

        if not self._output_value_range in ['01','101', '0255']:
            raise NotImplementedError()

        if not self._output_data_type in ['uint8', 'f32', 'f64']:
            raise NotImplementedError()

        if not self._output_channel_place in ['cl']:
            raise NotImplementedError()

    

        if ('a' in self._input_color_mode) and (not 'a' in self._output_color_mode):
            self._do_discard_alfa = True
        else:
            self._do_discard_alfa = False
        # if (self._input_color_mode == 'bgr') and (self._output_color_mode == 'rgb'):
        #     self._do_discard_alfa = True
        
        # 'cc_rs' = centre crop for aspect ratio then recale
        # 'cp_rs' = centre pad for aspect ratio then recale
        # 'nc_rs' = no crop for aspect ratio, only recale
        # 'cc_cp' = centre crop or pad for correct size but no recale
        if not self.format_crop in ['cc_rs', 'cp_rs', 'nc_rs', 'cc_cp']:
            raise NotImplementedError()

    #TODO this is the unique function the rest could be moved to parent class
    def convert(self, array_image, metadata):
        if (array_image is None):
            raise Exception('array_image cannot be none')

        if (self._input_data_type == 'uint8') and  (array_image.dtype !=np.uint8):
            raise Exception('incorrect array dtype: ' + str(array_image.dtype))
        if (self._input_data_type == 'f32') and  (array_image.dtype !=np.float32):
            raise Exception('incorrect array dtype: ' + str(array_image.dtype))
        if (self._input_data_type == 'f64') and  (array_image.dtype !=np.float64):
            raise Exception('incorrect array dtype: ' + str(array_image.dtype))

        # for jpg this reads the image in uint8 0-255 channel last
        if self._do_discard_alfa:
            if array_image.shape[2] == 4:
                array_image = array_image[:,:,0:3]

        metadata['list_size_source'] = [array_image.shape[0], array_image.shape[1]]
        
        # crop to aspect ratio if a list size was provided
        if self.list_size_output:
            value_pad = (127,)
            if self.format_crop == 'nc_rs':
                pass
            elif self.format_crop == 'cc_rs':
                # bring the bigger dimension down to match aspect_target
                aspect_current = array_image.shape[0] / float(array_image.shape[1])
                aspect_target = self.list_size_output[0] / float(self.list_size_output[1])
                if aspect_target < aspect_current:
                    list_size_cropped = [int(array_image.shape[1] * aspect_target), array_image.shape[1]]
                else:
                    list_size_cropped = [array_image.shape[0], int(array_image.shape[0] / aspect_target)]
                array_image = ToolsArrayImage.croppad_centre(array_image, list_size_cropped, value_pad)
                
            elif self.format_crop == 'cp_rs':
                # brings the smaller dimension up to match aspect_target
                aspect_current = array_image.shape[0] / float(array_image.shape[1])
                aspect_target = self.list_size_output[0] / float(self.list_size_output[1])
                if aspect_target < aspect_current:
                    list_size_cropped = [array_image.shape[0], int(array_image.shape[0] / aspect_target)]
                else:
                    list_size_cropped = [int(array_image.shape[1] * aspect_target), array_image.shape[1]]
                array_image = ToolsArrayImage.croppad_centre(array_image, list_size_cropped, value_pad)

            elif self.format_crop == 'cc_cp':
                raise NotImplementedError()
                #TODO rescale both
            else:
                raise NotImplementedError()
        
 

        metadata['list_size_cropped'] = [array_image.shape[0], array_image.shape[1]]
        offset_crop_0 = int((metadata['list_size_source'][0] - metadata['list_size_cropped'][0]) / 2)
        offset_crop_1 = int((metadata['list_size_source'][1] - metadata['list_size_cropped'][1]) / 2)
        metadata['list_offset_crop'] = [offset_crop_0, offset_crop_1]


  
        if self.list_size_output:
            metadata['list_ratio_rescale'] = [self.list_size_output[0] / array_image.shape[0], self.list_size_output[1] / array_image.shape[1]]
            array_image = ToolsArrayImage.rescale(array_image, self.list_size_output)
        else:
            metadata['list_ratio_rescale'] = [1.0, 1.0] 
        metadata['list_size_rescaled'] = [array_image.shape[0], array_image.shape[1]]

  
        # color mode
        if (self._input_color_mode[0] == self._output_color_mode[2]) and (self._input_color_mode[2] == self._output_color_mode[0]):
            array_image[:,:,[0, 1, 2]] = array_image[:,:,[2, 1, 0]]
  

        # images are now in uint8 0255 change data type earlier?

        if self._input_value_range != self._output_value_range:
            if self._input_value_range == '01':
                if self._output_value_range == '101':
                    array_image = (array_image * 2.0) - 1.0
                elif self._output_value_range == '0255':
                    array_image = array_image * 255
                else:
                    raise NotImplementedError()
            elif self._input_value_range == '0255':
                if self._output_value_range == '01':
                    array_image = array_image / 255
                elif self._output_value_range == '101':
                    array_image = ((array_image / 255.0) * 2) - 1.0
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError('input_value_range: ' + self._input_value_range +  '  output_value_range: ' + self._output_value_range )

        # change data type
        if self._input_data_type != self._output_data_type:
            if self._output_data_type == 'f64':
                array_image = array_image.astype(np.float64)
            if self._output_data_type == 'f32':
                array_image = array_image.astype(np.float32)
            elif self._output_data_type == 'uint8':
                array_image = array_image.astype(np.uint8)
            else:
                raise NotImplementedError()


        if self._output_channel_place != self._input_channel_place:
            array_image = np.moveaxis(array_image, 2, 0)


        if self.do_batch:
            array_image = np.expand_dims(array_image, axis=0)
        # print(json_post.keys())
        return array_image, metadata
