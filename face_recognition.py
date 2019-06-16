# 1. load the training data (numpy arrays of all persons)
		# x values are stored in numpy arrays
		# y values are assinged by us for each person
# 2. read the video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of the face(int)
# 5. map the predicted id to the username
# 6. display the prediction on the screen (bounding box and name)

import cv2
import numpy as np 
import os

############ KNN CODE ################################################
def distance(pt1, pt2):
	return np.sqrt(((pt1-pt2)**2).sum())


def knn(x , y, test, k=5):
	
	vals = []
	m = x.shape[0]

	for i in range(m):
		d = distance(test, x[i])
		vals.append([d, y[i]])

	vals = sorted(vals)
	# first k points
	vals = vals[:k]
	vals = np.array(vals)

	new_vals = np.unique(vals[:,1], return_counts = True)
	index = new_vals[1].argmax()
	pred = new_vals[0][index]

	return pred

######################################################################

#init camera
cap = cv2.VideoCapture(0)

#face detection 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0
names = {} #mapping b/w id and names

# Data preparation 
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		print("loaded"+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# create labels for class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)  

face_dataset = np.concatenate(face_data, axis = 0)
labels_dataset = np.concatenate(labels, axis = 0).reshape((-1, 1))

# Testing 

while True:
	
	ret, frame = cap.read()

	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for face in faces:
		x,y,w,h = face

		# get the face region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		out = knn(face_dataset, labels_dataset, face_section.flatten())

		# display name and rectangle around face
		pred_name = names[int(out)]
		cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 0), 2, cv2.LINE_AA)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

	cv2.imshow("Faces", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
