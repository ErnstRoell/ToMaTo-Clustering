import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt



X, y = make_blobs(n_samples=2000, centers=[(1,1),(10,10),(3,3)], n_features=3,random_state=0)
plt.scatter(X[:,0],X[:,1])
plt.show()

np.savetxt('C:/Users/gebruiker/documents/programming/python/tomato/test_data/toy_two_clusters.csv',X,
           delimiter=',')
