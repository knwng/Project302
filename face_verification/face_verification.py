'''
This is a class to calculate the similarity of two faces 
'''
from __future__ import print_function ,division

import os 
import sys
caffe_root = 'data/wangq/code/caffe-master/';
sys.path.insert(0,caffe_root+ 'python');
os.environ['GLOG_minloglevel'] = 2;
import caffe

import numpy as np 
import time 

class FaceVerfication:
	def __init__(self,model_proto, model_weight):
		self.net = caffe.Net(model_proto, model_weight, caffe.TEST);

	def vlen(self,x):
		return np.sqrt(x.dot(x));

	def similarity(self, face1, face2):
		'''
		Args:
		face1: cv2 image 
		face2: cv2 image 
		Return:
		a scalar in (0-1) to measure the similarity between face1 and face2 
		'''
		since = time.time();
		face1 = (face1 - 127.5) / 128.0;
		face2 = (face2 - 127.5) / 128.0;
		face1 = face1.transpose((2,0,1));
		face2 = face2.transpose((2,0,1));
		# TODO : here should be take carefully 
		o1 = self.net.forward(data = face1)['fc5'][0].copy();
		o2 = self.net.forward(data = face2)['fc5'][0].copy();
		cosdist = o1.dot(o2) / (self.vlen(o1) * self.vlen(o2) + 1.0e-5);
		print('TIME COSUMING {}'.format(time.time() - since));
		return cosdist;

		
