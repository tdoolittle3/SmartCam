import cv2 as cv
import imutils
from imutils.video import FPS
import sys
import time
from threading import Thread
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import dlib

if sys.version_info >= (3, 0):
	from queue import Queue
else:
	from Queue import Queue

lastNFrames = 4
ip = '192.168.0.11'
username = 'admin'
password = '########'
streamUrl = "rtsp://"+username+":"+password+"@"+ip+"/live/ch0"

## Frame parsing thread object
# ref www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv.VideoCapture(path)
		self.stopped = False

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return

				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

def detectMotion(frame, avg):
    # convert it to grayscale and blur it
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if avg is None:
        avg = gray.copy().astype("float")
        return ([], avg)
    cv.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv.absdiff(gray, cv.convertScaleAbs(avg))

    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    return (cnts, avg)

def adjustCamera(rois, width, height):
    if len(rois) == 0: return
    maxX = -1e200
    minX = 1e200
    maxY = -1e200
    minY = 1e200
    for (x,y,w,h) in rois:
        # Find outermost bounding boxes
        if x < minX: minX = x
        if x + w > maxX: maxX = x + w
        if y < minY : minY = y
        if y + h > maxY: maxY = y + h  
    #Determine direction

    delta = [ min([x[0] for x in rois]),
        min([y[1] for y in rois]),
        width - max([w[0]+w[2] for w in rois]),
        height - max([h[1]+h[3] for h in rois]) ]
    directions = ['left', 'up', 'right', 'down']
    if min(delta) < 50:
        moveCamera(directions[delta.index(min(delta))])
        moveCamera('stop')
    

def moveCamera(direction):
    if direction not in ['left', 'right', 'up', 'down', 'stop']: 
        print 'Invalid direction'
        return
    requests.post('http://'+ip+'/hy-cgi/ptz.cgi?cmd=ptzctrl&act='+direction, auth=HTTPDigestAuth(username,password))

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)


def main():
    print "Smart vision for home: ", streamUrl
    #vcap = cv.VideoCapture(0)
    vcap = FileVideoStream(streamUrl).start()
    
    if not vcap:
        print "No Video Capture found"
        exit()

    #hogPeopleDetector = cv.HOGDescriptor()
    #hogPeopleDetector.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )

    faceCascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    faceTracker = dlib.correlation_tracker()

    time.sleep(2.0)
    fps = FPS().start()
    avg = None
    

    minContourArea = 200
    resizeWidth = 500
    trackingFace = 0
    while(1):
        frame = vcap.read()
        # option to skip frames for processing
        for i in range(1, lastNFrames):
            frame = vcap.read()       
       
        if frame is None:
            print "Error grabbing frame"
            continue
        frame = imutils.resize(frame, width=resizeWidth)
        width = frame.shape[1]  
        height = frame.shape[0]
        
        #(cnts, avg) = detectMotion(frame, avg)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = []
        if trackingFace < 1:
            #found, w = hogPeopleDetector.detectMultiScale(gray, winStride=(8,8), padding=(32,32), scale=1.05)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(10, 10), flags = cv.CASCADE_SCALE_IMAGE)        
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                faceTracker.start_track(frame, dlib.rectangle( x-10, y-20, x+w+10, y+h+20))
                trackingFace = 1
        if trackingFace > 0:
            trackingQuality = faceTracker.update( frame )

            #If the tracking quality is good enough, determine the
            #updated position of the tracked region and draw the
            #rectangle
            if trackingQuality >= 5.75:
                tracked_position =  faceTracker.get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                faces = [(t_x, t_y, t_w, t_h)]
                cv.rectangle(frame, (t_x, t_y),
                                            (t_x + t_w , t_y + t_h),
                                            (0, 255, 0) ,2)
            else:
                trackingFace = 0

        adjustCamera(faces, width, height)



        cv.imshow('VIDEO', frame)
        cv.waitKey(1)
        fps.update()

    vcap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    print "Starting SmartCam with OpenCV ", cv.__version__
    print cv.getBuildInformation()
    main()








'''
   
'''




'''
width = frame.shape[1]  #vcap.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
height = frame.shape[0]  #vcap.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT)
        
currentlyMovingCamera = direction != 'stop'
if minX < (width / 4):
direction = 'left'
elif maxX > (width - (width / 4)):
direction = 'right'
elif minY < (height / 4):
direction = 'down'
elif maxY > (height - (height / 4)):
direction = 'up'
elif currentlyMovingCamera:
direction = 'stop'

if (direction != 'stop') or (currentlyMovingCamera and direction == 'stop'):
moveCamera(direction)

            for c in cnts:
            
                #print "Found movement with area ", cv.contourArea(c)
                if cv.contourArea(c) < minContourArea:
                    continue
            
                (x, y, w, h) = cv.boundingRect(c)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
'''

