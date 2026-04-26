import matplotlib.pyplot as plt
from math import log

mu_I = [log(y, 10) for y in [1e-5, 0.001, 1, 30, 100]]

plt.subplot(2, 2, 1)
plt.plot(mu_I, [30.74, 30.598 ,31.465, 30.767, 30.613], marker='o', linestyle='dashed', label='$m_G$=0%')
plt.plot(mu_I, [30.47, 30.37, 31.06, 30.35, 30.2], marker='o', linestyle='dashed', label='$m_G$=10%')
plt.plot(mu_I, [29.80, 29.75, 30.20, 29.58, 29.46], marker='o', linestyle='dashed', label='$m_G$=25%')
plt.plot(mu_I, [28.09, 28.118, 28.34, 27.995, 27.92], marker='o', linestyle='dashed', label='$m_G$=50%')
plt.xlabel('log( $\mu_I$[cm^2/Vs] )'), plt.ylabel('Efficiency [%]'), 
plt.title("Efficiency vs IB mobility for 5*$\sigma_{CI}$=$\sigma_{IV}$")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.plot(mu_I, [30.328, 30.88, 31.48, 32.33, 32.43], marker='o', linestyle='dashed', label='$m_G$=0%')
plt.plot(mu_I, [31.072, 31.67, 32.39, 33.14, 33.21], marker='o', linestyle='dashed', label='$m_G$=10%')
plt.plot(mu_I, [31.895, 32.53, 33.39, 33.81, 33.84], marker='o', linestyle='dashed', label='$m_G$=25%')
plt.plot(mu_I, [31.51, 31.979, 32.42, 32.434, 32.433], marker='o', linestyle='dashed', label='$m_G$=50%')
plt.xlabel('log( $\mu_I$[cm^2/Vs] )'), plt.ylabel('Efficiency [%]'), 
plt.title("Efficiency vs IB mobility for $\sigma_{CI}$=5*$\sigma_{IV}$")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.plot(mu_I, [29.899 ,29.91 ,29.86 ,29.04 ,28.98], marker='o', linestyle='dashed', label='$m_G$=0%')
plt.plot(mu_I, [29.47, 	29.68, 	29.88, 	29.2, 	29.15], marker='o', linestyle='dashed', label='$m_G$=10%')
plt.plot(mu_I, [28.65, 	28.87, 	29.06, 	28.58, 	28.54], marker='o', linestyle='dashed', label='$m_G$=25%')
plt.plot(mu_I, [27.04, 	27.24, 	27.37, 	27.15 ,	27.13 ], marker='o', linestyle='dashed', label='$m_G$=50%')
plt.xlabel('log( $\mu_I$[cm^2/Vs] )'), plt.ylabel('Efficiency [%]'), 
plt.title("Efficiency vs IB mobility for $\sigma_{CI}$=$\sigma_{IV}$=1e-13")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(mu_I, [38.34, 	38.05, 	39.00, 	39.05, 	39.03 ], marker='o', linestyle='dashed', label='$m_G$=0%')
plt.plot(mu_I, [38.33, 	38.21, 	38.73, 	38.67, 	38.64 ], marker='o', linestyle='dashed', label='$m_G$=10%')
plt.plot(mu_I, [37.12, 	37.13, 	37.42, 	37.32, 	37.29 ], marker='o', linestyle='dashed', label='$m_G$=25%')
plt.plot(mu_I, [33.82, 	33.89, 	34.05, 	33.98, 	33.95 ], marker='o', linestyle='dashed', label='$m_G$=50%')
plt.xlabel('log( $\mu_I$[cm^2/Vs] )'), plt.ylabel('Efficiency [%]'), 
plt.title("Efficiency vs IB mobility for $\sigma_{CI}$=$\sigma_{IV}$=5e-13")
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()

