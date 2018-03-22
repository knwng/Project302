import cv2
import os
from project302.project302 import Project302
import matplotlib.pyplot as plt 
from config import* 
# This is ur testing video path
video_dir = './videos/1.mov';
cap = v2.VideoCapture(video_dir);
'''
Here we initialize the project302
1. init the class 
2. set detector model 
3. set tracker model
4. set nms and confidence threshold
4. surveillance   
''' 
project = Project302(10,2);
project.init_detector(detector_rfcn+'.prototxt',detector_rfcn + '.caffemodel');
project.init_tracker(tracker_goturn+'.prototxt',detector_rfcn+ '.caffemodel');
project.SetNMS(0.7);
# initialize the video input and output 
frame_width = 640;
frame_height = 480;
output = cv2.VideoWriter('./results/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height));

def show_result(image,dets):
	image = image[:,:,(2,1,0)];
	for idnex in xrange(dets):
		bbox = dets[index];
		cv2.rectangle(image,(bbox[0,bbox[1]),(bbox[2],bbox[3]),(0,255,0),3);
	return image;
def demo():
	while(cap.isOpened()):
		ret,frame = cap.read();
		if(not ret):
			print('dalong log : frame read error');
		dets = project302.Surveillance(frame);
		image = show_result(image,dets);
		output.write(image);

	# release resources 
	cap.release();
	out.release();

if __name__ == '__main__':
	demo();
	
