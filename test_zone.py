from lib import detect_module
from lib import data_handler
import numpy as np
from keras import models

im_val = []
dummy = []
file_name = []
ground_truth_path = "/home/jpr0/project_2/project/DaimlerBenchmark/Data/SampleData/annotations"

server_flag = True
if server_flag:
    data_handler.load_data_general("/home/tkdrlf9202/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                   im_val, dummy, file_name, format='pgm', label=0, datasize=6744)
else:
    data_handler.load_data_general("/mnt/hgfs/Shared/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                      im_val, dummy, file_name, format='pgm', label=0, datasize=6744)
samples = im_val
count = 1

for idx, sample in enumerate(samples):
    print "====================Picture #"+str(count)+" Detection===================="
    print "FILE_NAME: "+str(file_name[idx])
    boxes = detect_module.generate_bounding_boxes(model=0, image=sample, file_name=file_name[idx], downscale=1.2, step=5,
                                                  min_height=96, min_width=48, grd_truth_path=ground_truth_path)
    print ""
    count += 1

