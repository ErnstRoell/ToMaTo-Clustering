import matplotlib.pyplot as plt
import numpy as np

N = 5000

r1 = np.random.normal(10, 1, N)
T1 = np.random.uniform(0, 2*np.pi, N)


r2 = np.random.normal(5, 1, N)
T2 = np.random.uniform(0, 2*np.pi, N)

x1 = r1*np.cos(T1)
y1 = r1*np.sin(T1)

x2 = r2*np.cos(T2)
y2 = r2*np.sin(T2)

idx1 = np.arange(0, N)
idx2 = np.arange(N, 2*N)

out1 = np.column_stack((idx1, x1, y1))
out2 = np.column_stack((idx2, x2, y2))
out = np.vstack((out1, out2))

np.savetxt('C:/Users/Gebruiker/Documents/Programming/Python/Tomato/test_data/CircularCluster.csv',out,delimiter=',')

# fig = plt.figure()
# plt.scatter(x1, y1, c='r')
# plt.scatter(x2, y2, c='b')
# plt.show()
