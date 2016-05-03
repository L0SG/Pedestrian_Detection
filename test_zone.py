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
sample = im_val[0]
print 'sample image shape' +str(sample.shape)
pyramid_list, scale_list = detect_module.pyramid(sample, downscale=1.5, min_height=96, min_width=48)
print 'image pyramid shape list & downscale factor list'
for i in range(0, len(pyramid_list)):
    print pyramid_list[i].shape, scale_list[i]

print 'windows of the pyramid image shape list'
boxes = detect_module.generate_bounding_boxes(model=0, image=sample, downscale=99, step=7, min_height=96, min_width=48)

