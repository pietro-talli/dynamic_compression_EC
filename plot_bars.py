import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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

#plt.figure()
#plt.plot(avg_q[indexes], performance[indexes])

#plt.plot(q_s[1:], [28.43,115.6,329.9,408.2,441.4,480.6])

#plt.legend(['variable','fixed'])

#plt.figure()
#p=0
#for i in indexes:
#    ax = plt.subplot(1,5,p+1)
#    ax.bar(q_s, dist_mat[i])
#    ax.set_title('beta: '+str(beta_s[i]))
#    p+=1
#    ax.set_ylim(top=1)
#plt.show()


rate_fixed = [1,2,3,4,5,6]

### FIGURE 1, test the model on the technical problem

perf_A_fixed = [20.194058442237797, 24.99214827082544, 27.61201437286083, 28.958544494459918, 29.821322810364947, 32.6512]

perf_AA = [20.194058442237797, 28.0053, 29.0162, 29.6654, 30.7636, 31.8770, 32.1434, 32.6512]
rate_AA = [1, 1.23, 1.56, 1.91, 2.53, 3.43, 5.20, 6]

perf_BA = [20.194058442237797, 25.3477, 26.5678, 27.3560, 27.6467, 31.3554,31.6131 ,32.6512]
rate_BA = [1, 1.2320945945945945, 1.481283422459893, 1.5572700296735904, 2.7012072434607646, 3.95132365499573, 4.9454450535108325 ,6]

perf_CA = [20.3487, 26.5689, 27.0579, 29.4583, 30.7343, 31.3622, 32.3304, 32.6512]
rate_CA = [0.04524886877828054, 1.2762376237623762, 1.5136499197063547, 2.4518292682926828, 3.2756634952428643, 5.0,5.862420382165605, 6]

### FIGURE 2, test the mdoel on the semantic problem

perf_B_fixed = [0.1838, 0.05621, 0.04076, 0.03579, 0.02363, 0.01214]

perf_AB = [0.1838, 0.1781, 0.0961, 0.0856, 0.0739, 0.0499, 0.0467, 0.01214]
rate_AB = [1, 1.23, 1.56, 1.91, 2.53, 3.43, 5.20, 6]

perf_BB = [0.1438, 0.1403, 0.0514, 0.051012, 0.041001, 0.0341, 0.0221, 0.01214]
rate_BB = [1, 1.0320945945945945, 1.481283422459893, 1.5572700296735904, 2.7012072434607646, 3.95132365499573, 4.9454450535108325 ,6]

perf_CB = [0.2965,0.1478,0.0611,0.0547,0.0457,0.0441,0.0341, 0.01214]
rate_CB = [0.04524886877828054, 1.1762376237623762, 1.5136499197063547, 2.4518292682926828, 3.2756634952428643, 5.0,5.862420382165605, 6]

### FIGURE 3, test the mdoel on the effectiveness problem (episode length)

perf_C_fixed = [31.09, 106.9, 325.4, 427.1, 465.7, 492.2]

perf_AC = [31.09, 58.9, 62.2, 88.6, 95.2, 312.3, 361.6, 492.2]
rate_AC = [1, 1.23, 1.56, 1.91, 2.53, 3.43, 5.20, 6]

perf_BC = [31.09, 58.4, 87.2, 146.0, 316.0, 419.7, 466.0, 492.2]
rate_BC = [1, 1.0320945945945945, 1.481283422459893, 1.5572700296735904, 2.7012072434607646, 3.95132365499573, 4.9454450535108325 ,6]

perf_CC = [22.1, 303.0, 328.0, 399.4, 435.9, 469.9, 481.0, 492.2]
rate_CC = [0.04524886877828054, 1.1762376237623762, 1.5136499197063547, 2.4518292682926828, 3.2756634952428643, 4.9012,5.862420382165605, 6]

### FIGURE 4, test the mdoel on the effectiveness problem (angle RMS error)

perf_C_fixed_rms_angle = [0.04511221596990319, 0.04310921261816572, 0.037216631121842536, 0.002120792022203152, 0.0008126434449312531, 0.0004314920138979987]

perf_AC_rms_angle = [0.07886026614254088, 0.05839160219900393, 0.05073462513044807, 0.04411221596990319, 0.023109278139379602, 0.005258412014236308, 0.0004714920138979987,0.0004314920138979987]
perf_BC_rms_angle = [0.10112487810409752, 0.04911221596990319, 0.014098066409611035 , 0.010072126939660316, 0.006413453529101213, 0.006335409505244653, 0.001723256497743983, 0.0004314920138979987]
perf_CC_rms_angle = [0.008124552873004637, 0.0029415309392780173, 0.0018484732876190894, 0.0011010334125407231, 0.001078159577779067, 0.0008883516585675265, 0.00047247973659052196,0.0004314920138979987]

### FIGURE 5, test the mdoel on the effectiveness problem (position RMS error)

perf_C_fixed_rms_position = [0.06413665697049578,0.35224688460063613, 1.6359298579860841, 3.8238404479998236,4.766684805941564,5.193876253280737]

perf_AC_rms_position = [ 0.06413665697049578, 0.41160390927511825, 0.9828463453052683, 1.7400087588561517, 1.9358602857378393, 3.754093584035746, 4.597668732856841, 5.193876253280737]
perf_BC_rms_position = [ 0.06413665697049578, 0.4018180964083476,   1.2266157412017018, 1.2394542675943119, 1.4147326839329013,  4.529034183553184,  4.619422345398644, 5.193876253280737]
perf_CC_rms_position = [0.048442670276776525, 2.911149892556571, 3.223191716625234, 3.6087695080387974, 3.631512186269963, 4.861854543295313, 5.1624293190587975 , 5.193876253280737]

### FIGURE 6, histograms level A

### FIGURE 7, histograms level B

### FIGURE 8, histograms level C

### PLOTS
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


fig1 = plt.figure()

plt.step(rate_fixed, perf_A_fixed, where = 'post')
plt.step(rate_AA, perf_AA, where = 'post')
plt.step(rate_BA, perf_BA, where = 'post')
plt.step(rate_CA, perf_CA, where = 'post')
plt.xlim(0,6.5)
plt.ylabel('PSNR (dB)')
plt.xlabel('Bits per feature')
plt.title('Results in the image reconstruction task')
plt.legend(['fixed','level A','level B', 'level C'])

tikzplotlib_fix_ncols(fig1)
plt.savefig('../figures/Image_Reconstuction.png')

import tikzplotlib
tikzplotlib.save("../figures/Image_Reconstuction.tex")

fig2 = plt.figure()

plt.step(rate_fixed, perf_B_fixed, where = 'post')
plt.step(rate_AB, perf_AB, where = 'post')
plt.step(rate_BB, perf_BB, where = 'post')
plt.step(rate_CB, perf_CB, where = 'post')
plt.xlim(0,6.5)
plt.ylabel('MSE')
plt.xlabel('Bits per feature')
plt.title('Results in the state estimation task')
plt.legend(['fixed','level A','level B', 'level C'])
plt.savefig('../figures/State_Estimation.png')

tikzplotlib_fix_ncols(fig2)
tikzplotlib.save("../figures/State_Estimation.tex")

fig3 = plt.figure()

plt.step(rate_fixed, perf_C_fixed, where = 'post')
plt.step(rate_AC, perf_AC, where = 'post')
plt.step(rate_BC, perf_BC, where = 'post')
plt.step(rate_CC, perf_CC, where = 'post')
plt.xlim(0,6.5)
plt.ylabel('Average episode length')
plt.xlabel('Bits per feature')
plt.title('Results in the control task')
plt.legend(['fixed','level A','level B', 'level C'])
plt.savefig('../figures/Episode_Length.png')

tikzplotlib_fix_ncols(fig3)
tikzplotlib.save("../figures/Episode_Length.tex")

fig4 = plt.figure()

plt.step(rate_fixed, perf_C_fixed_rms_angle, where = 'post')
plt.step(rate_AC, perf_AC_rms_angle, where = 'post')
plt.step(rate_BC, perf_BC_rms_angle, where = 'post')
plt.step(rate_CC, perf_CC_rms_angle, where = 'post')
plt.xlim(0,6.5)
plt.ylabel('RMS angle')
plt.xlabel('Bits per feature')
plt.title('Results in the control task (RMS angle)')
plt.legend(['fixed','level A','level B', 'level C'])
plt.savefig('../figures/RMS_angle.png')

tikzplotlib_fix_ncols(fig4)
tikzplotlib.save("../figures/RMS_angle.tex")

fig5 = plt.figure()

plt.step(rate_fixed, perf_C_fixed_rms_position, where = 'post')
plt.step(rate_AC, perf_AC_rms_position, where = 'post')
plt.step(rate_BC, perf_BC_rms_position, where = 'post')
plt.step(rate_CC, perf_CC_rms_position, where = 'post')
plt.xlim(0,6.5)
plt.ylabel('RMS position')
plt.xlabel('Bits per feature')
plt.title('Results in the control task (RMS position)')
plt.legend(['fixed','level A','level B', 'level C'])
plt.savefig('../figures/RMS_position.png')

tikzplotlib_fix_ncols(fig5)
tikzplotlib.save("../figures/RMS_position.tex")

plt.show()