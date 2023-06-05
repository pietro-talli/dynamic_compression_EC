import numpy as np
import matplotlib.pyplot as plt

performance = np.array([479.09, 450.94, 269.22, 362.69, 19.21])
avg_q = np.array([5.000939280719698, 4.596709096553865, 1.1541118787608646, 2.4808514158096444, 0.052056220718375845])
q_s = [0,1,2,3,4,5,6]
beta_s = [0.001, 0.01, 0.15, 0.07, 0.2]

dist_mat = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.9990607192803023, 0.0009392807196977604],
            [0.0, 0.0, 0.0, 0.0, 0.4032909034461347, 0.5967090965538653, 0.0],
            [0.7052967832999034, 0.0, 0.0, 0.028415422331178962, 0.26628779436891764, 0.0, 0.0],
            [0.46386721442554246, 0.0, 0.0, 0.00953982740081061, 0.1807328572610218, 0.3458601009126251, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

indexes = np.argsort(beta_s)

plt.figure()
plt.plot(avg_q[indexes], performance[indexes])

plt.plot(q_s[1:], [28.43,115.6,329.9,408.2,441.4,480.6])

plt.legend(['variable','fixed'])

plt.figure()
p=0
for i in indexes:
    ax = plt.subplot(1,5,p+1)
    ax.bar(q_s, dist_mat[i])
    ax.set_title('beta: '+str(beta_s[i]))
    p+=1
    ax.set_ylim(top=1)
plt.show()