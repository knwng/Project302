#/uer/bin/env python
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np 
import caffe

CLASSES = ('__background__','face')

'''
Detection class :
	input a cv2 image 
	output a np.array contains faces coordinates and scores  
'''
class Detector :
	
	def __init__(self,model_proto,model_weight,CONF_THRESH = 0.8,NMS_THRESH = 0.7):
		self.net = caffe.Net(model_proto,model_weight,caffe.TEST);
		self.conf_thresh = CONF_THRESH;
		self.nms_thresh = NMS_THRESH;
		# we only need face answer 
		self.cls_ind = 1;
		

	def Set_CONF(CONF):
		self.conf_thresh = CONF;
	def SET_NMS(NMS):
		self.nms_thresh = NMS;
	# Do confidence filter 
	def ConfFilter(self,dets,CONF_THRESH):
		keep = np.where(dets[:,-1] >= CONF_THRESH)[0];
		return dets[keep,:];
	# Do detection
	def detect(self,image):
		'''
		Args:
		image: cv2 image 
		Return:
		np.array with each row [bbox_leftupx,bbox_leftupy,bbox_rightdownx,bbox_rightdowny,score]
		'''
		scores,boxes = im_detect(self.net,im);
        	cls_boxes = boxes[:, 4* self.cls_ind:4*(self.cls_ind + 1)]
        	cls_scores = scores[:, self.cls_ind];
        	dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32);
		# CPU_NMS is much faster than GPU NMS when the number of boxes
		# is relatice small (e.g. , < 10k)
       		keep = nms(dets, NMS_THRESH,force_cpu =  True);
        	dets = dets[keep, :];
		dets = ConfFilter(dets,self.CONF_THRESH);
		return dets.copy();	
