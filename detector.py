from lib import detect_module
from lib import data_handler
from keras.models import model_from_json
import os
from operator import itemgetter


patchsize = (64, 32)
server_flag = True
t_set_path = '/home/tkdrlf9202/CaltechPedestrians/data-USA/images'
a_set_path = '/home/tkdrlf9202/CaltechPedestrians/data-USA/annotations'
# for bootstrapping
t_set_list = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
# for testing
#t_set_list = ['set06', 'set07', 'set08', 'set09', 'set10']
res_path = os.path.join(os.getcwd(), 'detection_result')
if not os.path.exists(res_path):
    os.mkdir(res_path)
# load model and compile
CNN_model_path = os.path.join(os.getcwd(), 'model.json')
CNN_weight_path = os.path.join(os.getcwd(), 'weights.h5')
model = model_from_json(open(CNN_model_path).read())
model.load_weights(CNN_weight_path)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

for set_id in xrange(0, len(t_set_list)):
    if not os.path.exists(os.path.join(res_path, t_set_list[set_id])):
        os.mkdir(os.path.join(res_path, t_set_list[set_id]))
    path = os.path.join(t_set_path, t_set_list[set_id])
    vid_list = [name for name in os.listdir(path)]
    vid_list.sort()
    for vid_id in xrange(0, len(vid_list)):
        result_file = open(os.path.join(os.getcwd(), res_path, t_set_list[set_id], vid_list[vid_id]+'_unsorted.txt'), 'w+')
        im_val = []
        dummy = []
        file_name = []
        vid_path = os.path.join(t_set_path, t_set_list[set_id], vid_list[vid_id])
        datasize = len([name for name in os.listdir(vid_path) if os.path.isfile(os.path.join(vid_path, name))])/5
        data_handler.load_data_general(vid_path, im_val, dummy, file_name, format='jpg', label=0, datasize=datasize)
        samples = im_val
        count = 1
        sum_fp_pi = 0
        sum_miss_rate_pi = 0

        for idx, sample in enumerate(samples):
            print "====================Picture #"+str(count)+" Detection===================="
            print "FILE_NAME: "+str(file_name[idx])
            grd_truth_path = os.path.join(a_set_path, t_set_list[set_id], vid_list[vid_id])
            fp_pi, miss_rate_pi = detect_module.generate_bounding_boxes(model=model, image=sample, file_name=file_name[idx],
                                                                        downscale=1.5, step=13,
                                                                        min_height=patchsize[0], min_width=patchsize[1],
                                                                        grd_truth_path=grd_truth_path, trainingFlag=0,
                                                                        min_prob=0.1, result_file=result_file,
                                                                        fp_name=str(t_set_list[set_id])+str(vid_list[vid_id]))
            sum_fp_pi += fp_pi
            sum_miss_rate_pi += miss_rate_pi
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
        result_sorted = open(os.path.join(os.getcwd(), res_path, t_set_list[set_id], vid_list[vid_id]+'.txt'), 'w+')
        for i in xrange(0, len(lines)):
            result_sorted.write(lines[i])