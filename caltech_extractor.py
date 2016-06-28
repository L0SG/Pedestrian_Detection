import os
from PIL import Image
import glob
from lib import data_handler
patchsize = (64, 32)
db = 'TudBrussels'
t_set_path = '/home/tkdrlf9202/CaltechPedestrians/data-'+db+'/images'
a_set_path = '/home/tkdrlf9202/CaltechPedestrians/data-'+db+'/annotations'
t_set_list = ['set00']
output_path = '/home/tkdrlf9202/CaltechPedestrians/data-'+db+'/images_cropped'
if not os.path.exists(output_path):
    os.mkdir(output_path)
for set_id in xrange(0, len(t_set_list)):
    path = os.path.join(t_set_path, t_set_list[set_id])
    vid_list = [name for name in os.listdir(path)]
    vid_list.sort()
    for vid_id in xrange(0, len(vid_list)):
        vid_path = os.path.join(t_set_path, t_set_list[set_id], vid_list[vid_id])
        for f in glob.glob(os.path.join(vid_path, "*.png")):
            count = 0
            basename = os.path.basename(str(f))[:-4]
            img = Image.open(str(f))
            g_truth = open(os.path.join(a_set_path, t_set_list[set_id], vid_list[vid_id], basename+'.txt'))
            lines = g_truth.readlines()[1:]
            """
            if len(lines) == 0:
                data_handler.extract_caltech_random_patches(f, name=str(t_set_list[set_id])+str(vid_list[vid_id])+str(basename),
                                                            patchsize=patchsize, datasize=10)
            """
            for line_idx in xrange(0, len(lines)):
                lines[line_idx] = lines[line_idx].split()
            for line_idx in xrange(0, len(lines)):
                tag = lines[line_idx][0]
                x_start = int(lines[line_idx][1])
                y_start = int(lines[line_idx][2])
                x_delta = int(lines[line_idx][3])
                y_delta = int(lines[line_idx][4])
                occluded = int(lines[line_idx][5])
                if tag == 'person' and y_delta >= 50 and occluded == 0:
                    img_cropped = img.crop((x_start, y_start, x_start+x_delta, y_start+y_delta))
                    img_cropped.save(os.path.join(output_path,
                                                  str(t_set_list[set_id])+str(vid_list[vid_id])+
                                                  str(basename)+str(count)+'.png'))
                    count += 1
