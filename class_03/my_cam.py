import cv2
import numpy as np

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print facec
# font = cv2.FONT_HERSHEY_SIMPLEX

data_collection = True
n = 10
skip_frame = 10
data = []

ix = 0
flag = False
while True:
	_, fr = rgb.read()

	gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	faces = facec.detectMultiScale(gray, 1.3, 5)
	ix += 1

	for (x,y,w,h) in faces:
		fc = fr[x:x+w, y:y+h, :]
		if data_collection:
			if ix%skip_frame == 0:
				roi = cv2.resize(fc, (64, 64))
				data.append(roi)
			if len(data) >= n:
				flag = True
				break
		cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
	if flag:
		break
	cv2.imshow('rgb', fr)
	cv2.imshow('gray', gray)
	if cv2.waitKey(1) == 27:
		break

print len(data)

data = np.asarray(data)
print data.shape

np.save('face_02', data)

cv2.destroyAllWindows()
