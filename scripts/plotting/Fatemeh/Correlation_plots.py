
import numpy as np
import matplotlib
matplotlib.use('Agg') # tells matplotlib not to load any gui display functions
import matplotlib.pyplot as plt
from os import path
from glob import glob
import pandas as pd
import yaml
#import matplotlib.pyplot.annotate as annotate
from simudo.example.fourlayer import sweep_extraction as se


#mu=1, sigma_ci=sigma_iv=1e-13
folder0 = 'JUL20a/1'
folder1 = 'JUL21a/3'
folder2 = 'JUL21a/4'
folder3 = 'JUL21a/5'
#mu=1, sigma_ci=sigma_iv=5e-13
folder4 = 'JUL22b/4'
folder5 = 'JUL22b/5'
folder6 = 'JUL22b/6'
folder7 = 'JUL22b/7'
#mu=1, sigma_ci=5*sigma_iv
folder8 = 'JUL21b/4'
folder9 = 'JUL21b/5'
folder10 = 'JUL21b/6'
folder11 = 'JUL21b/7'
#mu=1, 5*sigma_ci=sigma_iv
folder12 = 'JUL22a/4'
folder13 = 'JUL22a/5'
folder14 = 'JUL22a/6'
folder15 = 'JUL22a/7'

data0 = se.SweepData(folder0) # prefix = pd_I uses intensity based data instead
data1 = se.SweepData(folder1)
data2 = se.SweepData(folder2)
data3 = se.SweepData(folder3)
data4 = se.SweepData(folder4)
data5 = se.SweepData(folder5)
data6 = se.SweepData(folder6)
data7 = se.SweepData(folder7)
data8 = se.SweepData(folder8)
data9 = se.SweepData(folder9)
data10 = se.SweepData(folder10)
data11 = se.SweepData(folder11)
data12 = se.SweepData(folder12)
data13 = se.SweepData(folder13)
data14 = se.SweepData(folder14)
data15 = se.SweepData(folder15)

# extract spatial data at Max power point
spatial0 = data0.get_spatial_data(data0.mpp_row)
IB_mask0 = data0.IB_mask(spatial0)  

spatial1 = data1.get_spatial_data(data1.mpp_row)
IB_mask1 = data1.IB_mask(spatial1)  

spatial2 = data2.get_spatial_data(data2.mpp_row)
IB_mask2 = data2.IB_mask(spatial2)  

spatial3 = data3.get_spatial_data(data3.mpp_row)
IB_mask3 = data3.IB_mask(spatial3)  

spatial4 = data4.get_spatial_data(data4.mpp_row)
IB_mask4 = data4.IB_mask(spatial4) 

spatial5 = data5.get_spatial_data(data5.mpp_row)
IB_mask5 = data5.IB_mask(spatial5)

spatial6 = data6.get_spatial_data(data6.mpp_row)
IB_mask6 = data6.IB_mask(spatial6)

spatial7 = data7.get_spatial_data(data7.mpp_row)
IB_mask7 = data7.IB_mask(spatial7)

spatial8 = data8.get_spatial_data(data8.mpp_row)
IB_mask8 = data8.IB_mask(spatial8)

spatial9 = data9.get_spatial_data(data9.mpp_row)
IB_mask9 = data9.IB_mask(spatial9)

spatial10 = data10.get_spatial_data(data10.mpp_row)
IB_mask10 = data10.IB_mask(spatial10)

spatial11 = data11.get_spatial_data(data11.mpp_row)
IB_mask11 = data11.IB_mask(spatial11)

spatial12 = data12.get_spatial_data(data12.mpp_row)
IB_mask12 = data12.IB_mask(spatial12)

spatial13 = data13.get_spatial_data(data13.mpp_row)
IB_mask13 = data13.IB_mask(spatial13)

spatial14 = data14.get_spatial_data(data14.mpp_row)
IB_mask14 = data14.IB_mask(spatial14)

spatial15 = data15.get_spatial_data(data15.mpp_row)
IB_mask15 = data15.IB_mask(spatial15)


plt.plot([se.subgap_mismatch(spatial0, IB_mask0),se.subgap_mismatch(spatial1, IB_mask1),se.subgap_mismatch(spatial2, IB_mask2),
se.subgap_mismatch(spatial3, IB_mask3)],[29.859938300625183,29.87618791427941,29.063754588404594, 27.37490443060956], 
 marker='o', color = 'b', linestyle="None", label = '$r_{opt}$=1 (1e-13)')

plt.plot([se.subgap_mismatch(spatial4, IB_mask4), se.subgap_mismatch(spatial5, IB_mask5),se.subgap_mismatch(spatial6, IB_mask6),
se.subgap_mismatch(spatial7, IB_mask7)],[39.00426310271262, 38.72881368348476, 37.419197808394716, 34.054553179714603],
 marker='*', color = 'g',linestyle="None", label = '$r_{opt}$=1 (5e-13)')

plt.plot([se.subgap_mismatch(spatial8, IB_mask8),se.subgap_mismatch(spatial9, IB_mask9),se.subgap_mismatch(spatial10, IB_mask10),
se.subgap_mismatch(spatial11, IB_mask11)],[31.48153547160531, 32.389366854740065, 33.392238672119756, 32.42247512404745],
 marker='^', color = 'orange',linestyle="None", label = '$r_{opt}$=1/5')

plt.plot([se.subgap_mismatch(spatial12, IB_mask12),se.subgap_mismatch(spatial13, IB_mask13),se.subgap_mismatch(spatial14, IB_mask14),
se.subgap_mismatch(spatial15, IB_mask15)],[31.46515534310174, 31.056808645092654, 30.202734478853666, 28.338820383212393],
 marker='s', color = 'm',linestyle="None", label = '$r_{opt}$=5')

plt.xlabel('$m_L$'),plt.ylabel('$\eta$ [%]'), plt.title('Correlation between efficiency and local mismatch, $\mu_I$=1')
plt.legend(), plt.grid(True), plt.tight_layout()
plt.savefig('Eff_vs_LocalMismatch_MPP_mu=1.png')
plt.clf()

#mu=30, sigma_ci=sigma_iv=1e-13
folder0 = 'JUL20a/2'
folder1 = 'JUL21a/6'
folder2 = 'JUL21a/7'
folder3 = 'JUL21a/8'
#mu=30, sigma_ci=sigma_iv=5e-13
folder4 = 'JUL22b/8'
folder5 = 'JUL22b/9'
folder6 = 'JUL22b/10'
folder7 = 'JUL22b/11'
#mu=30, sigma_ci=5*sigma_iv
folder8 = 'JUL21b/8'
folder9 = 'JUL21b/9'
folder10 = 'JUL21b/10'
folder11 = 'JUL21b/11'
#mu=30, 5*sigma_ci=sigma_iv
folder12 = 'JUL22a/8'
folder13 = 'JUL22a/9'
folder14 = 'JUL22a/10'
folder15 = 'JUL22a/11'

data0 = se.SweepData(folder0) # prefix = pd_I uses intensity based data instead
data1 = se.SweepData(folder1)
data2 = se.SweepData(folder2)
data3 = se.SweepData(folder3)
data4 = se.SweepData(folder4)
data5 = se.SweepData(folder5)
data6 = se.SweepData(folder6)
data7 = se.SweepData(folder7)
data8 = se.SweepData(folder8)
data9 = se.SweepData(folder9)
data10 = se.SweepData(folder10)
data11 = se.SweepData(folder11)
data12 = se.SweepData(folder12)
data13 = se.SweepData(folder13)
data14 = se.SweepData(folder14)
data15 = se.SweepData(folder15)

# extract spatial data at Max power point
spatial0 = data0.get_spatial_data(data0.mpp_row)
IB_mask0 = data0.IB_mask(spatial0)  

spatial1 = data1.get_spatial_data(data1.mpp_row)
IB_mask1 = data1.IB_mask(spatial1)  

spatial2 = data2.get_spatial_data(data2.mpp_row)
IB_mask2 = data2.IB_mask(spatial2)  

spatial3 = data3.get_spatial_data(data3.mpp_row)
IB_mask3 = data3.IB_mask(spatial3)  

spatial4 = data4.get_spatial_data(data4.mpp_row)
IB_mask4 = data4.IB_mask(spatial4) 

spatial5 = data5.get_spatial_data(data5.mpp_row)
IB_mask5 = data5.IB_mask(spatial5)

spatial6 = data6.get_spatial_data(data6.mpp_row)
IB_mask6 = data6.IB_mask(spatial6)

spatial7 = data7.get_spatial_data(data7.mpp_row)
IB_mask7 = data7.IB_mask(spatial7)

spatial8 = data8.get_spatial_data(data8.mpp_row)
IB_mask8 = data8.IB_mask(spatial8)

spatial9 = data9.get_spatial_data(data9.mpp_row)
IB_mask9 = data9.IB_mask(spatial9)

spatial10 = data10.get_spatial_data(data10.mpp_row)
IB_mask10 = data10.IB_mask(spatial10)

spatial11 = data11.get_spatial_data(data11.mpp_row)
IB_mask11 = data11.IB_mask(spatial11)

spatial12 = data12.get_spatial_data(data12.mpp_row)
IB_mask12 = data12.IB_mask(spatial12)

spatial13 = data13.get_spatial_data(data13.mpp_row)
IB_mask13 = data13.IB_mask(spatial13)

spatial14 = data14.get_spatial_data(data14.mpp_row)
IB_mask14 = data14.IB_mask(spatial14)

spatial15 = data15.get_spatial_data(data15.mpp_row)
IB_mask15 = data15.IB_mask(spatial15)


plt.plot([se.subgap_mismatch(spatial0, IB_mask0),se.subgap_mismatch(spatial1, IB_mask1),se.subgap_mismatch(spatial2, IB_mask2),
se.subgap_mismatch(spatial3, IB_mask3)], [29.036297763179636, 29.19642609586967, 28.580818909819117, 27.15016426237363], 
marker='o', color = 'b', linestyle="None", label = '$r_{opt}$=1 (1e-13)')

plt.plot([se.subgap_mismatch(spatial4, IB_mask4),se.subgap_mismatch(spatial5, IB_mask5),se.subgap_mismatch(spatial6, IB_mask6),
se.subgap_mismatch(spatial7, IB_mask7)],[39.05409218903253, 38.67042673647372, 37.316007695844655, 33.978389933993375],
 marker='*', color = 'g',linestyle="None", label = '$r_{opt}$=1 (5e-13)')

plt.plot([se.subgap_mismatch(spatial8, IB_mask8),se.subgap_mismatch(spatial9, IB_mask9),se.subgap_mismatch(spatial10, IB_mask10),
se.subgap_mismatch(spatial11, IB_mask11)],[32.334526541470493, 33.138969801783447, 33.814042840392466, 32.434237778437636], 
 marker='^', color = 'orange',linestyle="None", label = '$r_{opt}$=1/5')

plt.plot([se.subgap_mismatch(spatial12, IB_mask12),se.subgap_mismatch(spatial13, IB_mask13),se.subgap_mismatch(spatial14, IB_mask14),
se.subgap_mismatch(spatial15, IB_mask15)],[30.76737091106847, 30.345500683163806, 29.583433528506886, 27.995091864075],
 marker='s', color = 'm',linestyle="None", label = '$r_{opt}$=5')

plt.xlabel('$m_L$'),plt.ylabel('$\eta$ [%]'), plt.title('Correlation between efficiency and local mismatch, $\mu_I$=30')
plt.legend(), plt.grid(True), plt.tight_layout()
plt.savefig('Eff_vs_LocalMismatch_MPP_mu=30.png')
