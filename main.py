import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity

import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = "C:/Users/Gebruiker/Documents/Programming/Python/Tomato"
os.chdir(PATH)

#file = './test_data/crater_256.csv'
#file = './test_data/crater_10000.csv'
#file = './test_data/spirals.csv'
#file = './test_data/toy_two_clusters.csv'
#file = './test_data/test.csv'
#file = './test_data/CircularCluster.csv'
file = './test_data/iris.csv'

class HillClimbing():
    def __init__(self, file):
        self.kDensity = 6
        self.kGraph = 3
        self.tau = 3

#        self.cloud = pd.read_csv(file, delimiter=',', index_col=4)
        self.cloud = pd.read_csv(file, delimiter=',')
        self.label = self.cloud.index.values
        self.cloud = self.cloud.values
        self.parent = np.arange(0,self.cloud.shape[0])
        self.clusters = []
        
        self.compute_neighbours()
        self.compute_density()
        self.compute_parent()
        self.compute_persistence()
        self.clusterlabels = self.compute_cluster_labels()

    def compute_neighbours(self):
        nbrs = NearestNeighbors(n_neighbors=np.max([self.kDensity, self.kGraph])+1, 
                                algorithm='ball_tree').fit(self.cloud)
        self.dist, self.neighbours = nbrs.kneighbors(self.cloud)

    def compute_density(self):
#        self.density = KernelDensity(kernel='gaussian',
#                                     bandwidth=1).fit(self.cloud).score_samples(self.cloud)
        self.density = np.sqrt(self.kDensity / np.sum(self.dist[:,:self.kDensity+1]**2, axis=1))

    def compute_parent(self):
        for row in self.neighbours:
            MAX = np.argmax(self.density[row[:self.kGraph+1]])
            if MAX != 0:
                mask = self.parent == self.parent[row[0]]
                self.parent[mask] = self.parent[row[MAX]]

            
    def compute_cluster_labels(self):
        index = np.arange(0, len(self.parent))
        return self.parent[index == self.parent]

    def compute_persistence(self):
        for row in self.neighbours:
        #    if row[0] == 42:
        #        print(row)
        #        print(self.parent[row])
        #        print(self.density[row])

            ARGMAX = np.argmax(self.density[row[1:self.kGraph+1]])
            if ARGMAX != 0:
                for ii in row[1:]:
                    if self.density[ii] >= self.density[row[0]]:
                        if self.density[self.parent[ii]] < np.min([self.density[self.parent[row[0]]],
                                                                   self.tau + self.density[row[0]]]):
                            mask = self.parent == self.parent[ii]
                            self.parent[mask] = self.parent[row[0]]
        
    def visualize(self):
        labels = np.unique(self.parent)
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = colors[:len(labels)]

        color_map = np.vstack((labels, colors))
        color_dict = {label: color for (label, color) in color_map.T}
        label_color = [color_dict[str(l)] for l in self.parent]
        plt.scatter(self.cloud[:,0], self.cloud[:,1],color=label_color)
        plt.show()

H = HillClimbing(file)
H.visualize()

print(np.unique(H.parent, return_counts=True))
print(len(np.unique(H.parent)))









