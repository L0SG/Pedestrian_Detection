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


def pyramid(image, downscale, min_height, min_width):
    # input is 2D image (NOT 4d tensor)
    from skimage.transform import pyramid_gaussian
    import numpy as np

    pyramid_list = []
    # if downscale is 2, it halves the image
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=downscale)):
        if resized.shape[0] < min_height or resized.shape[1] < min_width:
            break
        pyramid_list.append(resized)
    pyramid_np = np.asarray(pyramid_list)
    return pyramid_np