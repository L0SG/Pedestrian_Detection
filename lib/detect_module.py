
"""
# obsolete
def sliding_window(images, shape, step):
    from theano import function
    from theano import tensor as T
    from theano.tensor.nnet.neighbours import images2neibs
    import numpy as np

    # theano function declaration
    t_images = T.tensor4('t_images')
    neibs = images2neibs(t_images, neib_shape=shape, neib_step=(step, step), mode='ignore_borders')
    window_function = function([t_images], neibs)

    # apply theano function to input images
    # outputs 2D tensor : [ index , FLATTENED patches ]
    output = window_function(images)

    # reshape output 2D tensor
    output_reshaped = np.reshape(output, (len(output), shape[0], shape[1]))
    return output_reshaped
"""

def pyramid(image, downscale, min_height, min_width):
    # input is 3D tensor (NOT 4d tensor)
    # input shape is (channels, height, width)
    from skimage.transform import pyramid_gaussian
    import numpy as np
    from theano import tensor as T
    from theano import function
    arr_tensor3 = T.tensor3('arr', dtype='float32')
    arr_shuffler = arr_tensor3.dimshuffle((1, 2, 0))
    shuffle_function = function([arr_tensor3], arr_shuffler)
    arr_tensor3_2 = T.tensor3('arr', dtype='float64')
    arr_deshuffler = arr_tensor3_2.dimshuffle((2, 0, 1))
    deshuffle_function = function([arr_tensor3_2], arr_deshuffler)

    image_shuffled = shuffle_function(image)

    pyramid_list = []
    scale_list = []
    scale_factor = 1
    # if downscale is 2, it halves the image
    for (i, resized) in enumerate(pyramid_gaussian(image_shuffled, downscale=downscale)):
        if resized.shape[0] < min_height or resized.shape[1] < min_width:
            break
        resized_deshuffled = deshuffle_function(resized)
        pyramid_list.append(resized_deshuffled)

        if i == 0:
            scale_list.append(scale_factor)
        else:
            scale_factor *= float(downscale)
            scale_list.append(scale_factor)

    return pyramid_list, scale_list


def generate_bounding_boxes(model, image, downscale, step, min_height, min_width):
    from skimage.util import view_as_windows
    import numpy as np
    from theano import tensor as T
    from theano import function
    arr_tensor4 = T.tensor4('arr', dtype='float32')
    arr_shuffler = arr_tensor4.dimshuffle((1, 0, 2, 3))
    shuffle_function = function([arr_tensor4], arr_shuffler)

    boxes = []

    pyramid_list, scale_list = pyramid(image, downscale, min_height, min_width)

    for i in xrange(0, len(pyramid_list)):
        window_list = []
        window_channels = []
        for channels in xrange(0, len(pyramid_list[i])):
            window = view_as_windows(pyramid_list[i][channels], window_shape=(min_height, min_width), step=step)
            window_reshaped = np.reshape(window, newshape=(window.shape[0]*window.shape[1], min_height, min_width))
            window_channels.extend(window_reshaped)
        window_list.append(window_channels)
        window_list = np.asarray(window_list)
        window_list = shuffle_function(window_list)
        # temporary code
        print window_list.shape
        boxes.append(window_list)
    # temporary return
    return boxes[0]