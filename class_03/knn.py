import numpy as np

addr = ['face_01.npy', 'face_02.npy']
data = np.zeros((20, 64, 64, 3))

data[:10] = np.load(addr[0])
data[10:] = np.load(addr[1])

data = data.reshape((20, -1))

print data.shape

def knn(data, testing, k=5):
    N = data.shape[0]
    dist = []
    
    for ix in range(N):
        d = distance(data[ix, :2], testing)
        dist.append([d, data[ix, -1]])
    
    neighbours = sorted(dist, key=lambda x: x[0])
    k_neighbours = neighbours[:k]
    
    dist_and_labels = np.array(k_neighbours)
    labels = dist_and_labels[:, -1]
    
    freq = np.unique(labels, return_counts=True)
    return freq[0][freq[1].argmax()]


def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

# print distance(np.array([1, 1]), np.array([0, 0]))