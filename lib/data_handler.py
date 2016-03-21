def load_daimler_data(dir, X, y):
    import glob
    from PIL import Image
    from keras.preprocessing import image
    import os

    x_pos = []
    x_neg = []
    for f in glob.glob(os.path.join(os.getcwd(), dir, "ped_examples", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x_pos.append(arr)
    for f in glob.glob(os.path.join(os.getcwd(), dir, "non-ped_examples", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        x_neg.append(arr)
    for i in x_pos:
        X.extend(i)
        y.append(1)
    for i in x_neg:
        X.extend(i)
        y.append(0)