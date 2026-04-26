import matplotlib.pyplot as plt
import numpy as np

eff = [0.24865396019126101, 0.2556802090377667, 0.2635963388097214, 0.2710952447762366, 0.2778051750193065, 
0.2836657753064048, 0.2887093390404493, 0.29299124638881013, 0.29656632902177144, 0.29948402880816366,
0.3017897313829831, 0.30353002779490573, 0.3047580089881941, 0.30553294490945426 , 0.30591594415364476, 
0.3059651849631789, 0.3057323698913894, 0.3052611867840578, 0.30458731804021255, 0.30373923914810974]

thick = list(np.arange(0.1, 2.1, 0.1))

plt.plot(thick, eff, linestyle = '--', marker ='o')
plt.xlabel('IB thickness [$\mu$m]'), plt.ylabel('$\eta$'), plt.title("Efficiency VS IB thickness, GCM=0, 5*$\sigma_{CI}$ = $\sigma_{IV}$, $\mu_I$ = 0.001")
plt.plot(thick[eff.index(max(eff))], max(eff), 'ro')
plt.grid(True)
plt.tight_layout()
plt.savefig("Eff_VS_thickness_GCM=0_mu=0.001_5sigmaCI=sigmaIV.png")
plt.show()

