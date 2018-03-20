import detection.detection as detection
import tracker.tracker as tracker
import cv2
import numpy as np
class Project302:
	def __init__(self,detect_interval,max_face,show_result = False):
		print('init Project302\n');
		self.detect_interval = detect_interval;
		self.max_face = max_face;
		self.frame = 0;
		self.tracker = None;
		self.detector = None;
		self.show_result =show_result;
	def SetNMS(NMS):
		self.nms_threshold = NMS;
	def SetCONf(CONF):
		self.conf_threshold = CONF;
	def init_detection(self,model_proto,model_weight):
		self.detector = detection.Detection(model_proto,model_weight);
	def init_tracker(self,model_proto,model_weight):
		self.tracker = tracker.Tracker(model_proto,model_weight);
	def Surveillance(self,image):
		'''
		This is an interface for surveillance project
		
		it receives an image as an input and output bboxes for 
		
		'''
		self.frame = self.frame + 1;
		if(self.frame % detect_interval == 0):
			#detect
			dets = self.detector(image);
			bboxes = dets[:,:-1];
			scores = dets[:,-1];
			self.tracker.UpdateImageCache(image);
			self.tracker.UpdateBBoxCache(bboxes);  
		else:
			bboxes = self.tracker(image);
		bboxes = Filter(image,bboxes);
		if self.show_result :
			Show_result(image,bboxes);
		
		# currently we only 
		return bboxes;
	def NMS(self,dets,thresh):
		"""Pure Python NMS baseline."""
    		x1 = dets[:, 0]
    		y1 = dets[:, 1]
    		x2 = dets[:, 2]
    		y2 = dets[:, 3]

    		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    		order = scores.argsort()[::-1]

   		 keep = []
    		while order.size > 0:
        	i = order[0]
        	keep.append(i)
        	xx1 = np.maximum(x1[i], x1[order[1:]])
        	yy1 = np.maximum(y1[i], y1[order[1:]])
        	xx2 = np.minimum(x2[i], x2[order[1:]])
        	yy2 = np.minimum(y2[i], y2[order[1:]])

        	w = np.maximum(0.0, xx2 - xx1 + 1)
        	h = np.maximum(0.0, yy2 - yy1 + 1)
        	inter = w * h
        	ovr = inter / (areas[i] + areas[order[1:]] - inter)

        	inds = np.where(ovr <= thresh)[0]
        	order = order[inds + 1]

    		return keep
	def Filter(self,image,bboxes):
		'''
		In this filter 
		we do conf filter 
		we do nms filter 
		'''
		# first do conf filter ,now we dont have confidence output for tracker ,so here we dont do conf filter for now 
		
		# Do NMS Filter 
		# for tracking ,we just use python implementation of nms 
		# for detection, we use c version of nms 
		keep = NMS(bboxes,self.nms_threshold);
		return bboxes[keep,:];
			
		
	def Show_result(self,image,bboxes):
		import matplotlib.pyplot as plt;
		im = im[:,:,(2,1,0)];
		for index in xrange(bboxes):
			bbox = bboxes[index];
			plt.cla();
			plt.imshow(image);
			plt.gca().add_patch(
				plt.Rectangle((bbox[0],bbox[1]),
					       bbox[2] - bbox[0],
					       bbox[3] - bbox[1],
					       fill =  False;
					       edgecolor = 'g',
					       linewidth = 3);
			);
			plt.show();

