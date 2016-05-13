from lib import detect_module
from lib import data_handler
import numpy as np
from keras import models

im_val = []
dummy = []
server_flag = True
if server_flag:
    data_handler.load_data_general("/home/jpr1/project/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                   im_val, dummy, format='pgm', label=0, datasize=6744)
else:
    data_handler.load_data_general("/mnt/hgfs/Shared/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                      im_val, dummy, format='pgm', label=0, datasize=6744)
samples = im_val
for idx, sample in enumerate(samples):
    boxes = detect_module.generate_bounding_boxes(model=0, image=sample, file_name=str(idx), downscale=1.5, step=7,
                                                  min_height=96, min_width=48)

