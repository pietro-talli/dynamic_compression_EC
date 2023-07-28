import matplotlib.pyplot as plt
import numpy as np

# Plot the actions of the transmitter

betas_A = [2,1.5,1,0.5,0.1]
dist_A = [[0.7201783723522854, 0.0, 0.0, 0.0, 0.2798216276477146, 0.0, 0.0],
          [0.5878220140515222, 0.0, 0.0, 0.0, 0.41217798594847777, 0.0, 0.0],
          [0.4894433781190019, 0.0, 0.0, 0.0, 0.06621880998080615, 0.44433781190019195, 0.0],
          [0.31851115215229436, 0.0, 0.0, 0.0, 0.0, 0.6814888478477057, 0.0],
          [0.0, 0.0, 0.0, 0.013099930442847206, 0.30106654300950614, 0.0, 0.6858335265476466]

]

import tikzplotlib
for i in range(len(betas_A)):
    plt.figure()
    plt.bar(np.arange(7),dist_A[i])
    plt.title(r'$\beta = $'+str(betas_A[i]))
    plt.xlabel('Bits sent')
    plt.ylabel('Distribution')
    plt.ylim([0,1])
    tikzplotlib.save('../figures/level_A_distribution_'+str(betas_A[i])+'.tex')


betas_B = [0.01,0.0075,0.005,0.0025,0.001]
dist_B = [[0.6743295019157088, 0.07407407407407407, 0.01277139208173691, 0.02554278416347382, 0.21328224776500637, 0.0, 0.0],
          [0.6010309278350515, 0.021649484536082474, 0.01134020618556701, 0.0, 0.36597938144329895, 0.0, 0.0],
          [0.6315429353404037, 0.0, 0.0, 0.010605542251111872, 0.17824153267191242, 0.17960998973657202, 0.0],
          [0.17587490692479524, 0.0, 0.0, 0.0, 0.2786299329858526, 0.40565897244973936, 0.1398361876396128],
          [0.0, 0.0, 0.0, 0.0, 0.2742747542555742, 0.5269719491728603, 0.19875329657156557]]

import tikzplotlib
for i in range(len(betas_B)):
    plt.figure()
    plt.bar(np.arange(7),dist_B[i])
    plt.title(r'$\beta = $'+str(betas_B[i]))
    plt.xlabel('Bits sent')
    plt.ylabel('Distribution')
    plt.ylim([0,1])
    tikzplotlib.save('../figures/level_B_distribution_'+str(betas_B[i])+'.tex')

betas_C = [0.15,0.1,0.07,0.05,0.01]
dist_C = [[0.7055188578734285, 0.0, 0.0, 0.03643724696356275, 0.25804389516300874, 0.0, 0.0],
          [0.4307667421546425, 0.3185053380782918, 0.0, 0.0, 0.09899708832093174, 0.15173083144613395, 0.0],
          [0.30269132794329384, 0.0, 0.0, 0.0, 0.14298371912725663, 0.5543249529294495, 0.0],
          [0.451988360814743, 0.0, 0.0, 0.012470555632534294, 0.2287654149923791, 0.3067756685603436, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.13649279480414045, 0.8635072051958596]]

import tikzplotlib
for i in range(len(betas_C)):
    plt.figure()
    plt.bar(np.arange(7),dist_C[i])
    plt.title(r'$\beta = $'+str(betas_C[i]))
    plt.xlabel('Bits sent')
    plt.ylabel('Distribution')
    plt.ylim([0,1])
    tikzplotlib.save('../figures/level_C_distribution_'+str(betas_C[i])+'.tex')


plt.show()