def pyramid(__image, downscale, min_height, min_width):
    from PIL import Image
    from keras.preprocessing import image

    img = image.array_to_img(__image, scale=False)
    pyramid_list = []
    scale_list = []
    scale_factor = float(1)
    width_original = img.size[0]
    height_original = img.size[1]
    while True:
        width_new = int(float(width_original) / scale_factor)
        wpercent = float(width_new) / float(width_original)
        height_new = int((float(height_original) * float(wpercent)))
        if width_new < min_width or height_new < min_height:
            break
        img_downscaled = img.resize(size=(width_new, height_new), resample=Image.ANTIALIAS)
        arr = image.img_to_array(img_downscaled)
        pyramid_list.append(arr)
        scale_list.append(scale_factor)
        scale_factor *= downscale
    return pyramid_list, scale_list


def classify_windows_with_CNN(window_list, window_pos_list, CNN_model_path, CNN_weight_path, accuracy=0.9):
    from keras.models import Sequential
    from keras.models import model_from_json
    import keras.layers as layers
    from keras import callbacks
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from lib import data_handler
    import argparse
    #temporary module
    from keras.preprocessing import image
    import os


    #temporary path declaration
    CNN_model_path = os.path.join(os.getcwd(), 'model.json')
    CNN_weight_path = os.path.join(os.getcwd(), 'weights.h5')

    model = model_from_json(open(CNN_model_path).read())
    model.load_weights(CNN_weight_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print "CNN model is built."
    # We can also use probability
    print window_list.shape
    proba = model.predict_proba(window_list, batch_size=32) #Need to fix batch_size
	
    #temporary code
    print "CNN Probability List"
    print "length: "+str(len(proba))
    print proba



    CNN_detected_image_pos_list=[]
    CNN_detected_image_list=[]

    print "Print position of CNN detected image"
    for i in range(0, len(proba)):
        #proba[i][0]: probability of T, proba[i][1] proba. of F
        if proba[i][0] >= accuracy:
            print window_pos_list[i]
            CNN_detected_image_pos_list.append(window_pos_list[i])
            CNN_detected_image_list.append(window_list[i])

    CNN_detected_image_pos_list = np.asarray(CNN_detected_image_pos_list)

    # temporary code
    CNN_detected_image_list = np.asarray(CNN_detected_image_list)


    return CNN_detected_image_pos_list, CNN_detected_image_list


def cal_window_position(scale_list, xy_num_list, min_height, min_width, step):
    import numpy as np
    win_pos_list = []
    if len(scale_list) != len(xy_num_list):
        print "Something Wrong"

    # temporary code for debugging
    print "Print x, y number of window list per scale"
    print xy_num_list

    for x in range(0, len(scale_list)):
        scale_factor = scale_list[x]
        x_num = xy_num_list[x][0]
        y_num = xy_num_list[x][1]

        for i in range(0, y_num):
            y1 = i*step*scale_factor
            y2 = y1+scale_factor*min_height
            y1_int = int(y1)
            y2_int = int(y2)

            # temporary code for debugging
            if y2 > 480:
                print "Y range error"

            for j in range(0, x_num):
                x1 = j*step*scale_factor
                x2 = x1+scale_factor*min_width
                x1_int = int(x1)
                x2_int = int(x2)

                # temporary code for debugging
                if x2 > 640:
                    print "x range error"

                win_pos_list.append([x1_int, y1_int, x2_int, y2_int])
    win_pos_list = np.asarray(win_pos_list)
    return win_pos_list


def non_max_suppression_fast(boxes, overlapThresh):
    # source : http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Malisiewicz et al.
    # if there are no boxes, return an empty list
    import numpy as np
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def draw_rectangle(boxes_pos, __image):
    from PIL import ImageDraw
    from keras.preprocessing import image

    img = image.array_to_img(__image, scale=False)
    draw = ImageDraw.Draw(img)

    for box_pos in boxes_pos:
        pos_tuple = [(box_pos[0], box_pos[1]), (box_pos[2], box_pos[3])]
        draw.rectangle(pos_tuple, fill=None, outline='white')
    del draw
    img.save("Detected_Image.png", "PNG")

def extract_fp_examples(file_name, ground_truth, boxes, boxes_pos, accThresh=0.5):
    from PIL import Image
    from keras.preprocessing import image
    import os
    # if there are no boxes, return an empty list
    import numpy as np

    print "Test extract_fp_examples module "+str(len(boxes_pos))

    if len(boxes_pos) == 0:
        return []

    pick=[0] * len(boxes_pos)

    #temporary code


    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes_pos.dtype.kind == "i":
        boxes_pos = boxes_pos.astype("float")

    # grab the coordinates of the bounding boxes
    x1 = boxes_pos[:, 0]
    y1 = boxes_pos[:, 1]
    x2 = boxes_pos[:, 2]
    y2 = boxes_pos[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # keep looping while some indexes still remain in the indexes
    # list
    for i in xrange(len(boxes_pos)):
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        for j in range(len(ground_truth)):
            one_gnd_picture=ground_truth[j]         #for efficiency
            xx1 = np.maximum(one_gnd_picture[0],x1[i] )
            yy1 = np.maximum(one_gnd_picture[1], y1[i])
            xx2 = np.minimum(one_gnd_picture[2], x2[i])
            yy2 = np.minimum(one_gnd_picture[3], y2[i])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area
            if(overlap[i] > accThresh):    #Ovelap should over accuracy threshold which is set to 0.5 in default.
                pick[i]=1

    count=1
    check=0
    for i in xrange(len(pick)):
        if pick[i]==0: #false positive case
            im=image.array_to_img(boxes[i])
            count += 1
            if check==0:
                directory=os.path.join(os.getcwd(),"false_positive_set")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                check=1
        im.save(os.path.join(os.getcwd(),"false_positive_set",file_name+str(count)+".ppm"),"ppm")

    print "False Positive Images are saved with the name '"+file_name+"'"

    return


def generate_bounding_boxes(model, image, downscale, step, min_height, min_width, file_name, overlapThresh=0.9):
    from skimage.util import view_as_windows
    import numpy as np
    from theano import tensor as T
    from theano import function

    arr_tensor4 = T.tensor4('arr', dtype='float32')
    arr_shuffler = arr_tensor4.dimshuffle((1, 0, 2, 3))
    shuffle_function = function([arr_tensor4], arr_shuffler)

    boxes = []

    pyramid_list, scale_list = pyramid(image, downscale, min_height, min_width)

    total_window_list = []
    xy_num_list = []

    for i in xrange(0, len(pyramid_list)):
        window_channels = []
        for channels in xrange(0, len(pyramid_list[i])):
            window = view_as_windows(pyramid_list[i][channels], window_shape=(min_height, min_width), step=step)
            # window.shape[0]= number of columns of windows, window.shape[1]= the number of rows of windows
            xy_num_list.append([window.shape[1], window.shape[0]])
            # min_height & min_width info. should be discarded later for memory saving
            window_reshaped = np.reshape(window, newshape=(window.shape[0]*window.shape[1], min_height, min_width))
            # position of window must be calculated here : window_pos_list
            window_channels.append(window_reshaped)
        # window_list for one pyramid picture
        window_list = np.asarray(window_channels)

        # (3, n, H, W) to (n, 3, H, W)
        window_list = shuffle_function(window_list)

        # temporary code
        print window_list.shape
        total_window_list.extend(window_list)

    total_window_list = np.asarray(total_window_list)

    # classification
    total_window_pos_list = cal_window_position(scale_list, xy_num_list, min_height, min_width, step)

    # temporary code
    print "Window position list test"
    print "length: "+str(len(total_window_pos_list))
    print total_window_pos_list
    print "length of total window list: "+str(len(total_window_list))

    #temporary overlapThresh value
    overlapThresh=1
    accuracy=0.5

    CNN_box_pos_list, CNN_box_list = classify_windows_with_CNN(total_window_list, total_window_pos_list, 'temporary', 'path', accuracy)
    # NMS (overlap threshold can be modified)

    # temporary code
    print "CNN box position list"
    print "length:"+str(len(CNN_box_pos_list))
    print CNN_box_pos_list



    #temporary code    
    ground_truth=[[1,1,10,10]]

    extract_fp_examples(file_name, ground_truth, CNN_box_list, CNN_box_pos_list)

    sup_box_pos_list = non_max_suppression_fast(CNN_box_pos_list, overlapThresh)

    # temporary code
    print "Suppressed box position list"
    print "sup_box_pos_list length:"+str(len(sup_box_pos_list))
    print sup_box_pos_list


    draw_rectangle(sup_box_pos_list, image)

    # temporary return
    return sup_box_pos_list

# We get the PATH of ground_truth files, not position of ground_truth boxes position for encapsulation.





















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

"""
def pyramid(image, downscale, min_height, min_width):
    # input is 3D tensor (NOT 4d tensor)
    # input shape is (channels, height, width)
    from skimage.transform import pyramid_gaussian
    import numpy as np
    from theano import tensor as T
    from theano import function
    from skimage import img_as_float

    arr_tensor3 = T.tensor3('arr', dtype='float32')
    arr_shuffler = arr_tensor3.dimshuffle((1, 2, 0))
    shuffle_function = function([arr_tensor3], arr_shuffler)

    arr_tensor3_2 = T.tensor3('arr', dtype='float64')
    arr_deshuffler = arr_tensor3_2.dimshuffle((2, 0, 1))
    deshuffle_function = function([arr_tensor3_2], arr_deshuffler)

    tensor_float = T.tensor3('arr', dtype='float64')
    tensor_caster = T.cast(tensor_float, 'float32')
    cast_function = function([tensor_float], tensor_caster)

    image_shuffled = shuffle_function(image)

    pyramid_list = []
    scale_list = []
    scale_factor = 1
    # if downscale is 2, it halves the image
    for (i, resized) in enumerate(pyramid_gaussian(image_shuffled, downscale=downscale)):
        if resized.shape[0] < min_height or resized.shape[1] < min_width:
            break
        resized = cast_function(resized)
        resized_deshuffled = deshuffle_function(resized)
        pyramid_list.append(resized_deshuffled)

        if i == 0:
            scale_list.append(scale_factor)
        else:
            scale_factor *= float(downscale)
            scale_list.append(scale_factor)

    return pyramid_list, scale_list
"""


