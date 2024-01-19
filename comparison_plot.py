import matplotlib.pyplot as plt
import numpy as np

level_A = [[1, 1.23, 1.56, 1.91, 2.53, 3.43, 5.2, 6], [31.09, 58.9, 62.2, 88.6, 95.2, 312.3, 361.6, 492.2]]
    
level_B = [[1, 1.03, 1.48, 1.55, 2.7, 3.95, 4.94, 6], [31.09, 58.4, 87.2, 146, 316, 419.7, 466, 492.2]]

level_C = [[1.17, 1.51, 2.45, 3.27, 4.9, 5.86, 6], [303, 328, 399.4, 435.9, 469.9, 481.0, 492.2]]

digital = [[61.39, 128.25, 182.48], [32.65, 142.9, 201.5]]

semantic = [[]]

neural = [[20.253, 44.76, 136.2], [35.71, 160.34 ,463.5]]

plt.figure()

#level_A = 8*np.array(level_A)
#level_B = 8*np.array(level_B)
#level_C = 8*np.array(level_C)
digital = np.array(digital)
neural = np.array(neural)

plt.step(np.log10(level_A[0]), level_A[1], where='post', label='Level A')
plt.step(np.log10(level_B[0]), level_B[1], where='post', label='Level B')
plt.step(np.log10(level_C[0]), level_C[1], where='post', label='Level C')
plt.step(np.log10(digital[0]), digital[1], where='post', label='Digital')
plt.step(np.log10(neural[0]/3), neural[1], where='post', label='Neural')
plt.xlabel('log_10(l) [B]')
plt.ylabel('Episolde length [steps]')

plt.legend()

plt.show()
