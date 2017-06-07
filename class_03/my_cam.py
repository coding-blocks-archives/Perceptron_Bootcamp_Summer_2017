import cv2
import numpy as np


rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print facec
font = cv2.FONT_HERSHEY_SIMPLEX

data_collection = False
n = 10
skip_frame = 10
data = []

if not data_collection:
	addr = ['face_01.npy', 'face_02.npy']
	data = np.zeros((20, 64, 64, 3))

	data[:10] = np.load(addr[0])
	data[10:] = np.load(addr[1])

	all_data = data.reshape((20, -1))
	print all_data.shape
	train_data = np.zeros((20, all_data.shape[1]+1))
	train_data[:, :-1] = all_data
	train_data[:10, -1] = 1

def knn(data, testing, k=5):
    N = data.shape[0]
    dist = []
    
    for ix in range(N):
        d = distance(data[ix, :-1], testing)
        dist.append([d, data[ix, -1]])
    
    neighbours = sorted(dist, key=lambda x: x[0])
    k_neighbours = neighbours[:k]
    
    dist_and_labels = np.array(k_neighbours)
    labels = dist_and_labels[:, -1]
    
    freq = np.unique(labels, return_counts=True)
    return freq[0][freq[1].argmax()]


def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())


ix = 0
flag = False
while True:
	_, fr = rgb.read()

	gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	faces = facec.detectMultiScale(gray, 1.3, 5)
	ix += 1
	fac = None
	
	for (x,y,w,h) in faces:
		fc = fr[y:y+h, x:x+w, :]
		# cv2.imshow('face', fc)
		# print fc.shape, fr.shape
		if data_collection:
			if ix%skip_frame == 0:
				roi = cv2.resize(fc, (64, 64))
				data.append(roi)
			if len(data) >= n:
				flag = True
				break
		else:
			# Face recognition
			print "-"*100
			roi = cv2.resize(fc, (64, 64))
			# print train_data.shape
			out = str(knn(train_data, roi.flatten()))
			labels = {
				'1.0': 'XYZ',
				'0.0': 'ABC'
			}
			cv2.putText(fr, labels[out], (x, y), font, 1, (255, 255, 0), 2)
			# print 'Prediction:', 
		cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
		fac = fc
	if flag:
		break
	cv2.imshow('rgb', fr)
	# cv2.imshow('gray', gray)
	if cv2.waitKey(1) == 27:
		break

print len(data)

if data_collection:
	data = np.asarray(data)
	print data.shape, '--------------'

	np.save('face_02', data)

cv2.destroyAllWindows()
