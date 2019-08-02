# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--cascade", required=True,
#help="path to where the face cascade resides")
#ap.add_argument("-m", "--model", required=True,
#help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
help="path to the (optional) video file")
args = vars(ap.parse_args())


# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()	

	# if we are viewing a video and we did not grab a frame, then we
	# have reached the end of the video
	if args.get("video") and not grabbed:
		break
	# resize the frame, convert it to grayscale, and then clone the
	# original frame so we can draw on it later in the program	
	frame = imutils.resize(frame, width=300)

	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	
	
	facecascade = cv2.CascadeClassifier('/home/anurag/Books/Cuurently_Reading/smile_detect/haarcascade_frontalface_default.xml')
    	eye_cascade = cv2.CascadeClassifier('/home/anurag/Books/Cuurently_Reading/smile_detect/haarcascade_eye.xml')

	faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5)
 
    	print(’Total number of Faces found’,len(faces))
    
    	for (x, y, w, h) in faces:
		face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
		roi_gray = imgtest[y:y+h, x:x+w]
		roi_color = imgtest[y:y+h, x:x+w]        plt.imshow(face_detect)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
		    eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
		    plt.imshow(eye_detect)

			
	# show our detected faces along with smiling/not smiling labels
	#cv2.imshow("Face", frameClone)

	# if the ’q’ key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()


