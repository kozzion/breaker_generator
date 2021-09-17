import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

class ToolsArrayImage:

    @staticmethod
    def rgb_to_gray(array_source, list_weight=[0.299, 0.587, 0.114]):
        array_target = np.zeros((array_source.shape[0],array_source.shape[1], 1), array_source.dtype)
        for i in range(3):
            array_target[:,:,0] += array_source[:, :, i] * list_weight[i]
        return array_target

    @staticmethod
    def rotate(array_source, radian, reshape=False):
        degree = np.degrees(radian)
        return ndimage.rotate(array_source, degree, reshape=reshape)

    @staticmethod
    def transform_create_matrix_scale_rotate_translate(
        scale_x, scale_y, 
        radian_rotation, rotation_origen_x, rotation_origen_y,
        translate_x, translate_y):

        matrix_transform = np.eye(3)
        
        matrix_scale = np.eye(3)
        matrix_scale[0,0] = scale_x
        matrix_scale[1,1] = scale_y

        #TODO rotate around point in source image
        matrix_rotation = np.eye(3)
        matrix_rotation[0,0] = np.cos(radian_rotation)
        matrix_rotation[0,1] = -1 * np.sin(radian_rotation)
        matrix_rotation[1,0] = np.sin(radian_rotation)
        matrix_rotation[1,1] = np.cos(radian_rotation)



        matrix_translate = np.eye(3)
        matrix_translate[2,0] = translate_x
        matrix_translate[2,1] = translate_y
        
        matrix_transform = np.matmul(matrix_transform, matrix_scale)
        matrix_transform = np.matmul(matrix_transform, matrix_rotation)
        matrix_transform = np.matmul(matrix_transform, matrix_translate)
        return matrix_transform

    @staticmethod
    def transform_array_point(matrix_transfrom, array_x_source, array_y_source):
        count_point = array_x_source.shape[0]
        matrix_source = np.vstack([array_x_source, array_y_source, np.ones(count_point)]).T
        matrix_target = np.matmul(matrix_source, matrix_transfrom)
        array_x_target = matrix_target[:,0]
        array_y_target = matrix_target[:,1]
        return array_x_target, array_y_target


    @staticmethod
    def transform_array_image(matrix_transfrom, array_image_source, size_x_target, size_y_target, *, mode='linear'):
        if mode not in {'linear', 'cubic', 'quintic'}:
              raise Exception('unknown mode: ' + mode)
     
        if len(array_image_source.shape) == 2:
            array_image = np.expand_dims(array_image_source, axis=-1)
        else:
            array_image = array_image_source            
        count_channel = array_image_source.shape[2]

        shape_target = (size_y_target, size_x_target, count_channel)
        array_target = np.zeros(shape_target, dtype=array_image_source.dtype)
        

        y_source = np.arange(0, array_image_source.shape[0], 1)
        x_source = np.arange(0, array_image_source.shape[1], 1)
       
        y_target = np.arange(0, array_target.shape[0], 1)
        x_target = np.arange(0, array_target.shape[1], 1)

        xx_target, yy_target = np.meshgrid(x_target, y_target) #fucking weird function
        
        yy_target = np.reshape(yy_target, -1)
        xx_target = np.reshape(xx_target, -1)

        xx_target, yy_target = ToolsArrayImage.transform_array_point(matrix_transfrom, xx_target, yy_target)

        for index_channel in range(count_channel):
            spline = RectBivariateSpline(y_source, x_source, array_image_source[:,:,index_channel])
            zz_target = spline.ev(yy_target, xx_target)
            zz_target = np.reshape(zz_target, (size_y_target, size_x_target))
            array_target[:,:,index_channel] = zz_target
            
        if len(array_image_source.shape) == 2:
            return np.squeeze(array_target)
        else:
            return array_target

    # https://docs.scipy.org/doc/scipy/reference/interpolate.html
    @staticmethod
    def resample(array_source, list_size_target, list_offset_target, mode='linear'):
        if mode not in {'linear', 'cubic', 'quintic'}:
              raise Exception('unknown mode: ' + mode)
        count_channel = 1
        if 2 < len(array_source.shape):
            count_channel = array_source.shape[2]

        shape_target = (list_size_target[0], list_size_target[1], count_channel)
        array_target = np.zeros(shape_target, dtype=array_source.dtype)
        
        x_source = np.arange(0, array_source.shape[1], 1)
        y_source = np.arange(0, array_source.shape[0], 1)


        x_target = np.arange(0, array_target.shape[1], 1) * array_source.shape[1]/shape_target[1]
        y_target = np.arange(0, array_target.shape[0], 1) * array_source.shape[0]/shape_target[0]

        for index_channel in range(count_channel):
            if count_channel == 1:
                f = interpolate.interp2d(x_source, y_source, array_source[:,:], kind=mode)
            else:
                f = interpolate.interp2d(x_source, y_source, array_source[:,:,index_channel], kind=mode)
            array_target[:,:,index_channel] = f(x_target, y_target)

          

        if 2 == len(array_source.shape):
            return np.squeeze(array_target)
        else:
            return array_target

    # https://docs.scipy.org/doc/scipy/reference/interpolate.html
    @staticmethod
    def rescale(array_source, list_size, mode='linear'):
        if mode not in {'linear', 'cubic', 'quintic'}:
              raise Exception('unknown mode: ' + mode)
        count_channel = 1
        if 2 < len(array_source.shape):
            count_channel = array_source.shape[2]

        shape_target = (list_size[0], list_size[1], count_channel)
        array_target = np.zeros(shape_target, dtype=array_source.dtype)
        
        x_source = np.arange(0, array_source.shape[1], 1)
        y_source = np.arange(0, array_source.shape[0], 1)


        x_target = np.arange(0, array_target.shape[1], 1) * array_source.shape[1]/shape_target[1]
        y_target = np.arange(0, array_target.shape[0], 1) * array_source.shape[0]/shape_target[0]

        for index_channel in range(count_channel):
            if count_channel == 1:
                f = interpolate.interp2d(x_source, y_source, array_source[:,:], kind=mode)
            else:
                f = interpolate.interp2d(x_source, y_source, array_source[:,:,index_channel], kind=mode)
            array_target[:,:,index_channel] = f(x_target, y_target)

          

        if 2 == len(array_source.shape):
            return np.squeeze(array_target)
        else:
            return array_target

    @staticmethod
    def croppad_centre(array_image_source, list_size_target, value_pad):
        list_size_source = [array_image_source.shape[0], array_image_source.shape[1]]
        list_size_crop = [min(list_size_source[0], list_size_target[0]), min(list_size_source[1], list_size_target[1])]
        list_offset_source = [int((list_size_source[0] - list_size_crop[0]) / 2), int((list_size_source[1] - list_size_crop[1]) / 2)]
        list_offset_target = [int((list_size_target[0] - list_size_crop[0]) / 2), int((list_size_target[1] - list_size_crop[1]) / 2)]
        return ToolsArrayImage.croppad_offset(array_image_source, list_size_target, list_offset_source, list_offset_target, value_pad)


    @staticmethod
    def uncroppad_centre(array_image_target, list_size_source, value_pad):
        pass #offset
        list_size_target = [array_image_target.shape[0], array_image_target.shape[1]]
        list_size_crop = [min(list_size_source[0], list_size_target[0]), min(list_size_source[1], list_size_target[1])]
        list_offset_source = [int((list_size_source[0] - list_size_crop[0]) / 2), int((list_size_source[1] - list_size_crop[1]) / 2)]
        list_offset_target = [int((list_size_target[0] - list_size_crop[0]) / 2), int((list_size_target[1] - list_size_crop[1]) / 2)]
        return ToolsArrayImage.uncroppad_offset(array_image_target, list_size_source, list_offset_source, list_offset_target, value_pad)


    @staticmethod
    def croppad_offset(array_image_source, list_size_target, list_offset_source, list_offset_target, value_pad):
        if len(array_image_source.shape) == 2:
            is_mask = True
            array_image_target = np.expand_dims(array_image_source, axis=2)
        else:
            is_mask = False

        count_channel = 1
        if 2 < len(array_image_source.shape):
            count_channel = array_image_source.shape[2]

        array_image_target = np.zeros((list_size_target[0], list_size_target[1], count_channel), dtype=array_image_source.dtype)
        array_image_target[:,:,:] = value_pad
        if array_image_target.shape[0] <= list_offset_target[0] or array_image_target.shape[1] <= list_offset_target[1]:
            # destination out of scope
            return array_image_target

        if array_image_source.shape[0] <= list_offset_source[0] or array_image_source.shape[1] <= list_offset_source[1]:
            # source out of scope
            return array_image_target

        os_0 = list_offset_source[0]
        os_1 = list_offset_source[1]
        ot_0 = list_offset_target[0]
        ot_1 = list_offset_target[1]
        si_0 = min(array_image_source.shape[0] - list_offset_source[0], array_image_target.shape[0] - list_offset_target[0])
        si_1 = min(array_image_source.shape[1] - list_offset_source[1], array_image_target.shape[1] - list_offset_target[1])

        # copy data into target
        if count_channel == 1:
            array_image_target[ot_0:ot_0+si_0, ot_1:ot_1+si_1,0] = array_image_source[os_0:os_0+si_0, os_1:os_1+si_1]
        else:
            array_image_target[ot_0:ot_0+si_0, ot_1:ot_1+si_1, :] = array_image_source[os_0:os_0+si_0, os_1:os_1+si_1, :]
        if is_mask:
            return array_image_target[:, :, 0]
        else:
            return array_image_target

    @staticmethod
    def uncroppad_offset(array_image_target, list_size_source, list_offset_source, list_offset_target, value_pad):
        if len(array_image_target.shape) == 2:
            is_mask = True
            array_image_target = np.expand_dims(array_image_target, axis=2)
        else:
            is_mask = False

        array_image_source = np.zeros((list_size_source[0], list_size_source[1], array_image_target.shape[2]), dtype=array_image_target.dtype)
        array_image_source[:,:,:] = value_pad
        
        if array_image_source.shape[0] <= list_offset_source[0] or array_image_source.shape[1] <= list_offset_source[1]:
            # source out of scope
            return array_image_source

        if array_image_target.shape[0] <= list_offset_target[0] or array_image_target.shape[1] <= list_offset_target[1]:
            # destination out of scope
            return array_image_source

        os_0 = list_offset_source[0]
        os_1 = list_offset_source[1]
        ot_0 = list_offset_target[0]
        ot_1 = list_offset_target[1]
        si_0 = min(array_image_source.shape[0] - list_offset_source[0], array_image_target.shape[0] - list_offset_target[0])
        si_1 = min(array_image_source.shape[1] - list_offset_source[1], array_image_target.shape[1] - list_offset_target[1])

        # copy data into target
        array_image_source[os_0:os_0+si_0, os_1:os_1+si_1, :] = array_image_target[ot_0:ot_0+si_0, ot_1:ot_1+si_1, :]
        if is_mask:
            return array_image_source[:, :, 0]
        else:
            return array_image_source

 