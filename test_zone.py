from lib import detect_module
from lib import data_handler
import numpy as np
from keras import models
from keras.models import Sequential
from keras.models import model_from_json
import os
from operator import itemgetter

im_val = []
dummy = []
file_name=[]
result_file = open(os.path.join(os.getcwd(), "result.txt"), 'w+')
server_flag = True
ground_truth_path="/home/tkdrlf9202/data-Daimler_Extracted/annotations"
mode = 'L'
if server_flag:
    data_handler.load_data_general("/home/tkdrlf9202/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                   im_val, dummy, file_name, format='pgm', label=0, datasize=6744, mode=mode)
else:
    data_handler.load_data_general("/mnt/hgfs/Shared/DaimlerBenchmark/Data/TrainingData/NonPedestrians",
                                      im_val, dummy, file_name, format='pgm', label=0, datasize=6744, mode=mode)

# load model and compile
CNN_model_path = os.path.join(os.getcwd(), 'model.json')
CNN_weight_path = os.path.join(os.getcwd(), 'weights.h5')

model = model_from_json(open(CNN_model_path).read())
model.load_weights(CNN_weight_path)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

samples = im_val
count = 1
sum_fp_pi = 0
sum_miss_rate_pi = 0

for idx, sample in enumerate(samples):
    print "====================Picture #"+str(count)+" Detection===================="
    print "FILE_NAME: "+str(file_name[idx])
    fp_pi, miss_rate_pi = detect_module.generate_bounding_boxes(model=model, image=sample, file_name=file_name[idx],
                                                                downscale=1.2, step=11, min_height=64, min_width=32,
                                                                grd_truth_path=ground_truth_path, trainingFlag=1,
                                                                min_prob=0.1, fp_name='', result_file=result_file)
    sum_fp_pi+=fp_pi
    sum_miss_rate_pi+=miss_rate_pi
    print "FP per image: "+str(fp_pi)+"  Miss Rate per image: "+str(miss_rate_pi*100) + "%"
    count += 1
print ""
print "####################FINAL RESULT####################"
print "Avr FPPI: "+str(float(sum_fp_pi)/(count-1))+"   Avr MRPI: "+str(float(sum_miss_rate_pi)/(count-1))
print ""

# sort result file by frame number
result_file.seek(0)
lines = result_file.readlines()
lines = [i.split(',') for i in lines]
lines = sorted(lines, key=lambda x: int(itemgetter(0)(x)))
lines = [','.join(i) for i in lines]
result_sorted = open(os.path.join(os.getcwd(), "result_sorted.txt"), 'w+')
for i in xrange(0, len(lines)):
    result_sorted.write(lines[i])