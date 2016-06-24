def load_daimler_data(dir, X, y):
    import glob
    from PIL import Image
    from keras.preprocessing import image
    import os

    x_pos = []
    x_neg = []
    for f in glob.glob(os.path.join(dir, "Pedestrians", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x_pos.append(arr)
    for f in glob.glob(os.path.join(dir, "NonPedestrians", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x_neg.append(arr)
    for i in x_pos:
        X.extend(i)
        y.append(1)
    for i in x_neg:
        X.extend(i)
        y.append(0)


def load_daimler_detection_data(dir, X, y):
    import glob
    from PIL import Image
    from keras.preprocessing import image
    import os
    import numpy as np

    x = []
    for f in glob.glob(os.path.join(dir, "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x.append(arr)

    return np.asarray(x)


def load_data_general(dir, X, Y, file_name, format, label, datasize):
    # take all files from the specified path
    # and label each file with the specified label annotation
    # attach processed x, y to input list X, Y via extend
    # X, Y are needed to be converted to numpy format
    import glob
    from PIL import Image
    from keras.preprocessing import image
    import os

    x = []
    y = []
    file=[]
    for f in glob.glob(os.path.join(dir, "*."+str(format))):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x.append(arr)
        y.append(label)

        file_basename=os.path.basename(str(f))
        file.append(file_basename)
        if len(x) >= datasize:
            break

    X.extend(x)
    Y.extend(y)
    file_name.extend(file)

def load_data_train(dir, X, Y, file_name, format, label, patchsize, datasize):
    # take all files from the specified path
    # and label each file with the specified label annotation
    # attach processed x, y to input list X, Y via extend
    # X, Y are needed to be converted to numpy format


    import glob
    from PIL import Image
    from keras.preprocessing import image
    import os

    x = []
    y = []
    file=[]
    for f in glob.glob(os.path.join(dir, "*."+str(format))):
        img = Image.open(str(f))
        # resize images to fixed patchsize for training
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img.resize((patchsize[1], patchsize[0]), Image.ANTIALIAS)
        arr = image.img_to_array(img)
        x.append(arr)
        y.append(label)

        file_basename = os.path.basename(str(f))
        file.append(file_basename)
        if len(x) >= datasize:
            break

    X.extend(x)
    Y.extend(y)
    file_name.extend(file)


def load_data_random_patches(dir, X, Y, format, label, patchsize, datasize):
    # take all files from the specified path and generate random patches from these files
    # and label each file with the specified label annotation
    # attach processed x, y to input list X, Y via extend
    # X, Y are needed to be converted to numpy format
    import glob
    from PIL import Image
    from keras.preprocessing import image
    from sklearn.feature_extraction.image import extract_patches_2d
    import os
    import numpy as np
    from theano import tensor as T
    from theano import function
    # theano dimshuffle
    # greyscale only maybe
    # need to fix extending x if intende to use color channels too
    arr_tensor3 = T.tensor3('arr')
    arr_shuffler = arr_tensor3.dimshuffle((1, 2, 0))
    shuffle_function = function([arr_tensor3], arr_shuffler)

    arr_tensor3_2 = T.tensor3('arr')
    arr_deshuffler = arr_tensor3_2.dimshuffle(0, 'x', 1, 2)
    deshuffle_function = function([arr_tensor3_2], arr_deshuffler)

    x=[]
    y=[]
    while True:
        if len(x) >= datasize:
            break
        for f in glob.glob(os.path.join(dir, "*."+str(format))):
            img = Image.open(str(f))
            arr = image.img_to_array(img)
            arr_shuffled = shuffle_function(arr)
            patches = extract_patches_2d(arr_shuffled, patch_size=patchsize, max_patches=1)
            patches_deshuffled = deshuffle_function(patches)
            x.extend(patches_deshuffled)
            y.extend([label]*len(patches))
            if len(x) >= datasize:
                break
    X.extend(x)
    Y.extend(y)
