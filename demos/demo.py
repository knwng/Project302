import cv2
import os
import sys
import init_path
from project302 import Project302
import matplotlib.pyplot as plt 
from config import cfg

# This is ur testing video path
video_dir = './videos/1.mov';
cap = cv2.VideoCapture(video_dir);
'''
Here we initialize the project302
1. init the class 
2. set detector model 
3. set tracker model
4. set nms and confidence threshold
4. surveillance   
''' 
project = Project302(100,2);
project.init_tracker(cfg.tracker_goturn+'.prototxt',cfg.tracker_goturn+ '.caffemodel');
print('init tracker success');

#project.init_detector(cfg.detector_rfcn+'.prototxt',cfg.detector_rfcn + '.caffemodel');
#project.SetNMS(0.7);
print('dalong log : init success');
# initialize the video input and output 
frame_width = 640;
frame_height = 480;
#output = cv2.VideoWriter('./results/output.avi',cv2.cv.CV_FOURCC('M','J','P','G'), 10, (frame_width,frame_height));

def show_result(image,dets):
	image = image[:,:,(2,1,0)];
	for idnex in xrange(dets):
		bbox = dets[index];
		cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3);
	return image;
def demo():
	frame_index = 0;
	while(cap.isOpened()):
		ret,frame = cap.read();
		if(not ret):
			print('dalong log : frame read error');
		dets = project.Surveillance(frame);
		frame_index = frame_index + 1;
		print('dalong log : frame index {}'.format(frame_index));
		if(len(dets)==0):
			continue;
		image = show_result(image,dets);
#		output.write(image);

	# release resources 
	cap.release();
 #       output.release();

if __name__ == '__main__':
	demo();
	
