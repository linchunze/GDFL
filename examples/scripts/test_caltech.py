
#### Setup  #####

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage import data, exposure, img_as_float
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
#caffe_root = '/'  # this file is expected to be in {caffe_root}/examples
import os
#os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(2)
caffe.set_mode_gpu()
import pdb
import time
import shutil
###### Load labelMap ######

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'models/Caltech/labelmap.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
	num_labels = len(labelmap.item)
	labelnames = []
	if type(labels) is not list:
		labels = [labels]
	for label in labels:
		found = False
		for i in xrange(0, num_labels):
			if label == labelmap.item[i].label:
				found = True
				labelnames.append(labelmap.item[i].display_name)
				break
		assert found == True
	return labelnames


#### Load the net in the test phase for inference, and configure input preprocessing ####
model_def = 'models/Caltech/deploy.prototxt'
model_weights = 'models/Caltech/GDFL.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
				model_weights,  # contains the trained weights
				caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

#### Load an image. #####

# set net to batch size of 1
image_resize_h = 480
image_resize_w = 640
net.blobs['data'].reshape(1,3,image_resize_h,image_resize_w)

# read caltech test set
f = open('test_name.txt')
line = f.readline()

results_dir = 'output/'
if os.path.exists(results_dir):
	shutil.rmtree(results_dir)

if not os.path.exists(results_dir):
	os.makedirs(results_dir)


t0 = time.time()

while line:
	im_name = line[:-2]
	image = caffe.io.load_image('path/to/your/data/Caltech/test/images/'+ im_name +'.jpg')

	name_split = im_name.split('_', 3)
	setname = name_split[0]
	vname = name_split[1]
	iname = int(name_split[2][1:]) + 1

	if not os.path.exists(results_dir + setname):
		os.makedirs(results_dir + setname)

	dets_file_name = os.path.join(results_dir, setname, vname + '.txt')
	fid = open(dets_file_name, 'a')

	#### Run the net and examine the top_k results #### 
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image
	
	# Forward pass.
	detections = net.forward()['detection_out']

	# Parse the outputs.
	det_label = detections[0,0,:,1]
	det_conf = detections[0,0,:,2]
	det_xmin = detections[0,0,:,3]
	det_ymin = detections[0,0,:,4]
	det_xmax = detections[0,0,:,5]
	det_ymax = detections[0,0,:,6]

	# Get detections with confidence higher than 0.6.
	top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.05]

	top_conf = det_conf[top_indices]
	top_label_indices = det_label[top_indices].tolist()
	top_labels = get_labelname(labelmap, top_label_indices)
	top_xmin = np.round(det_xmin[top_indices]*image.shape[1])
	top_ymin = np.round(det_ymin[top_indices]*image.shape[0])
	top_xmax = np.round(det_xmax[top_indices]*image.shape[1])
	top_ymax = np.round(det_ymax[top_indices]*image.shape[0])
	
	for i in xrange(top_conf.shape[0]):
		w = top_xmax[i] - top_xmin[i] +1
		h = top_ymax[i] - top_ymin[i] +1
	
		score = top_conf[i]

		fid.write('%i %i %i %i %i %f\n' % (iname, top_xmin[i]+1, top_ymin[i]+1, w, h, score))
		
	line = f.readline()  
	fid.close()
	#print line
	

f.close()
fid.close()
t1 = time.time()
print str(t1-t0)	
