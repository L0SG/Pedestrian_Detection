from lib import detect_module
from lib import data_handler
import numpy as np
from keras import models
"""
print 'loading architecture and weights...'
model = models.model_from_json(open('model.json').read())
model.load_weights('weights.h5')
"""

im_val = []
dummy = []
data_handler.load_data_general("/mnt/hgfs/Shared/DaimlerBenchmark/Data/SampleData",
                                      im_val, dummy, format='pgm', label=0, datasize=1)
samples = im_val
count = 0
for sample in samples:
    boxes = detect_module.generate_bounding_boxes(model=0, image=sample, downscale=1.5, step=10,
                                                  min_height=96, min_width=48, img_count=count)
    count += 1
