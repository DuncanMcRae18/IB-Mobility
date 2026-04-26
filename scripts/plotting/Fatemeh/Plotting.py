# modified by Fatemeh MK (July 2021)

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

def IB_band_diagram(spatial, IB_mask):
    # copied from sweep_extraction but with some modifications to get rid of an error for the if statements.
    fig, Ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
    plt.plot(spatial["coord_x"], spatial["Ephi_CB"], color="k")
    plt.plot(spatial["coord_x"], (spatial["Ephi_VB"]), color="k")   
    plt.plot(
        spatial["coord_x"][IB_mask], spatial["Ephi_IB"][IB_mask], color="k"
    )
    plt.plot(
        spatial["coord_x"],
        (spatial["qfl_CB"]),
        color="blue",
        linestyle="--",
        label=r"$E_{F,C}$",
    )
    plt.plot(
        spatial["coord_x"][IB_mask],
        spatial["qfl_IB"][IB_mask],
        color="orange",
        linestyle="--",
        label=r"$E_{F,I}$",
    )
    plt.plot(
        spatial["coord_x"],
        (spatial["qfl_VB"]),
        color="red",
        linestyle="--",
        label=r"$E_{F,V}$",
    )
    plt.xlabel(r"x ($\mu$m)")
    plt.ylabel(r"Energy (eV)")
    plt.legend()

def subgap_generation_mismatch_diagram(spatial, IB_mask):
    #copied from sweep_exrtaction
    mismatch = (
        spatial["g_opt_ci_IB"] + spatial["g_opt_iv_IB"]
    )  # the CI term is always negative
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        mismatch[IB_mask],
        color="blue",
        label=r"$g_{iv}-g_{ci}$", #I changed this with the other label
    )
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        -mismatch[IB_mask],
        color="red",
        label=r"$g_{ci}-g_{iv}$",
    )
    plt.xlabel(r"X position ($\mu$m)")
    plt.ylabel(r"g (cm$^{-3}$s$^{-1}$)")
    plt.grid(True)
    plt.legend()

def jv_plot(df):
    p = plt.plot(df["v"], df["j"]/1000, marker='o', markersize=3, mec='k', mew=0.5)
    plt.xlabel(r"V (V)")
    plt.ylabel(r"J (mA/cm$^2$)")
    plt.grid(True)
    plt.ylim(-62, 0), plt.xlim(0, 1.4)
    return p

def IB_nr_recomb_diagram(spatial, IB_mask):
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        spatial["g_nr_top_IB"][IB_mask], #top is always positive since carrier enter IB
        color="green",
        label=r"$R_{ci}$",
    )
    plt.semilogy(
        spatial["coord_x"][IB_mask],
        -spatial["g_nr_bottom_IB"][IB_mask], #bottom is always negative since carrier leave IB
        color="red",
        label=r"-$R_{iv}$",
    )
    plt.xlabel(r"X position ($\mu$m)"), plt.ylabel(r"Recombination (cm$^{-3}$s$^{-1}$)")
    plt.legend(), plt.grid(True)

folder0 = 'JUL21b/0'
folder1 = 'JUL21b/4'
folder2 = 'JUL21b/8'
folder3 = 'JUL21b/12'
folder4 = 'JUL21b/1'
folder5 = 'JUL21b/5'
folder6 = 'JUL21b/9'
folder7 = 'JUL21b/13'
folder8 = 'JUL21b/2'
folder9 = 'JUL21b/6'
folder10 = 'JUL21b/10'
folder11 = 'JUL21b/14'
folder12 = 'JUL21b/3'
folder13 = 'JUL21b/7'
folder14 = 'JUL21b/11'
folder15 = 'JUL21b/15'

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

# #data at equilibrium
# dataeq1=se.SweepData(folder0, prefix='pd_I')
# dataeq2=se.SweepData(folder1, prefix='pd_I')
# dataeq3=se.SweepData(folder2, prefix='pd_I')
# dataeq4=se.SweepData(folder3, prefix='pd_I')
# spatialeq1 = dataeq1.get_spatial_data(dataeq1.v_row(0))
# IB_maskeq1 = dataeq1.IB_mask(spatialeq1)  
# spatialeq2 = dataeq2.get_spatial_data(dataeq2.v_row(0))
# IB_maskeq2 = dataeq2.IB_mask(spatialeq2) 
# spatialeq3 = dataeq3.get_spatial_data(dataeq3.v_row(0))
# IB_maskeq3 = dataeq3.IB_mask(spatialeq3)   
# spatialeq4 = dataeq4.get_spatial_data(dataeq4.v_row(0))
# IB_maskeq4 = dataeq4.IB_mask(spatialeq4)  

# #Plotting the filling fraction VS position in the IB 
# plt.figure(1)
# plt.plot(spatial0['coord_x'][IB_mask0], spatial0['u_IB'][IB_mask0]/(1e17), color = '#00aa00', label='$\mu_I$ = 0.001') 
# plt.plot(spatial1['coord_x'][IB_mask1], spatial1['u_IB'][IB_mask1]/(1e17), color = '#0011ff', label='$\mu_I$ = 1') 
# plt.plot(spatial2['coord_x'][IB_mask2], spatial2['u_IB'][IB_mask2]/(1e17), color = '#F5B326', label='$\mu_I$ = 30') 
# plt.plot(spatial3['coord_x'][IB_mask3], spatial3['u_IB'][IB_mask3]/(1e17), color = '#F526BC', label='$\mu_I$ = 100') 
# plt.plot(spatialeq1['coord_x'][IB_maskeq1], spatialeq1['u_IB'][IB_maskeq1]/(1e17), linestyle='dashed', color = '#00aa00', label='equilibrium') 
# plt.plot(spatialeq2['coord_x'][IB_maskeq2], spatialeq2['u_IB'][IB_maskeq2]/(1e17), linestyle='dashed', color = '#0011ff') 
# plt.plot(spatialeq3['coord_x'][IB_maskeq3], spatialeq3['u_IB'][IB_maskeq3]/(1e17), linestyle='dashed', color = '#F5B326') 
# plt.plot(spatialeq4['coord_x'][IB_maskeq4], spatialeq4['u_IB'][IB_maskeq4]/(1e17), linestyle='dashed', color = '#F526BC') 
# plt.title("f_I vs position for GCM = 0 at mpp, $\sigma_{ci}$ = $\sigma_{iv}$ = 1e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_GCM=0__sigmaCI=sigmaIV=1e-13_with_equilibrium.png")

# dataeq1=se.SweepData(folder4, prefix='pd_I')
# dataeq2=se.SweepData(folder5, prefix='pd_I')
# dataeq3=se.SweepData(folder6, prefix='pd_I')
# dataeq4=se.SweepData(folder7, prefix='pd_I')
# spatialeq1 = dataeq1.get_spatial_data(dataeq1.v_row(0))
# IB_maskeq1 = dataeq1.IB_mask(spatialeq1)  
# spatialeq2 = dataeq2.get_spatial_data(dataeq2.v_row(0))
# IB_maskeq2 = dataeq2.IB_mask(spatialeq2) 
# spatialeq3 = dataeq3.get_spatial_data(dataeq3.v_row(0))
# IB_maskeq3 = dataeq3.IB_mask(spatialeq3)   
# spatialeq4 = dataeq4.get_spatial_data(dataeq4.v_row(0))
# IB_maskeq4 = dataeq4.IB_mask(spatialeq4)

# plt.figure(2)
# plt.plot(spatial4['coord_x'][IB_mask4], spatial4['u_IB'][IB_mask4]/(1e17), color = '#00aa00', label='$\mu_I$ = 0.001') 
# plt.plot(spatial5['coord_x'][IB_mask5], spatial5['u_IB'][IB_mask5]/(1e17), color = '#0011ff', label='$\mu_I$ = 1') 
# plt.plot(spatial6['coord_x'][IB_mask6], spatial6['u_IB'][IB_mask6]/(1e17), color = '#F5B326', label='$\mu_I$ = 30') 
# plt.plot(spatial7['coord_x'][IB_mask7], spatial7['u_IB'][IB_mask7]/(1e17), color = '#F526BC', label='$\mu_I$ = 100') 
# plt.plot(spatialeq1['coord_x'][IB_maskeq1], spatialeq1['u_IB'][IB_maskeq1]/(1e17), linestyle='dashed', color = '#00aa00', label='equilibrium') 
# plt.plot(spatialeq2['coord_x'][IB_maskeq2], spatialeq2['u_IB'][IB_maskeq2]/(1e17), linestyle='dashed', color = '#0011ff') 
# plt.plot(spatialeq3['coord_x'][IB_maskeq3], spatialeq3['u_IB'][IB_maskeq3]/(1e17), linestyle='dashed', color = '#F5B326') 
# plt.plot(spatialeq4['coord_x'][IB_maskeq4], spatialeq4['u_IB'][IB_maskeq4]/(1e17), linestyle='dashed', color = '#F526BC') 
# plt.title("f_I vs position for GCM = 10 at mpp, $\sigma_{ci}$ = $\sigma_{iv}$ = 1e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_GCM=10__sigmaCI=sigmaIV=1e-13_with_equilibrium.png")

# dataeq1=se.SweepData(folder8, prefix='pd_I')
# dataeq2=se.SweepData(folder9, prefix='pd_I')
# dataeq3=se.SweepData(folder10, prefix='pd_I')
# dataeq4=se.SweepData(folder11, prefix='pd_I')
# spatialeq1 = dataeq1.get_spatial_data(dataeq1.v_row(0))
# IB_maskeq1 = dataeq1.IB_mask(spatialeq1)  
# spatialeq2 = dataeq2.get_spatial_data(dataeq2.v_row(0))
# IB_maskeq2 = dataeq2.IB_mask(spatialeq2) 
# spatialeq3 = dataeq3.get_spatial_data(dataeq3.v_row(0))
# IB_maskeq3 = dataeq3.IB_mask(spatialeq3)   
# spatialeq4 = dataeq4.get_spatial_data(dataeq4.v_row(0))
# IB_maskeq4 = dataeq4.IB_mask(spatialeq4)

# plt.figure(3)
# plt.plot(spatial8['coord_x'][IB_mask8], spatial8['u_IB'][IB_mask8]/(1e17),color = '#00aa00', label='$\mu_I$ = 0.001') 
# plt.plot(spatial9['coord_x'][IB_mask9], spatial9['u_IB'][IB_mask9]/(1e17), color = '#0011ff', label='$\mu_I$ = 1') 
# plt.plot(spatial10['coord_x'][IB_mask10], spatial10['u_IB'][IB_mask10]/(1e17), color = '#F5B326', label='$\mu_I$ = 30') 
# plt.plot(spatial11['coord_x'][IB_mask11], spatial11['u_IB'][IB_mask11]/(1e17), color = '#F526BC', label='$\mu_I$ = 100') 
# plt.plot(spatialeq1['coord_x'][IB_maskeq1], spatialeq1['u_IB'][IB_maskeq1]/(1e17), linestyle='dashed', color = '#00aa00', label='equilibrium') 
# plt.plot(spatialeq2['coord_x'][IB_maskeq2], spatialeq2['u_IB'][IB_maskeq2]/(1e17), linestyle='dashed', color = '#0011ff') 
# plt.plot(spatialeq3['coord_x'][IB_maskeq3], spatialeq3['u_IB'][IB_maskeq3]/(1e17), linestyle='dashed', color = '#F5B326') 
# plt.plot(spatialeq4['coord_x'][IB_maskeq4], spatialeq4['u_IB'][IB_maskeq4]/(1e17), linestyle='dashed', color = '#F526BC') 
# plt.title("f_I vs position for GCM = 25 at mpp, $\sigma_{ci}$ = $\sigma_{iv}$ = 1e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_GCM=25__sigmaCI=sigmaIV=1e-13_with_equilibrium.png")

# dataeq1=se.SweepData(folder12, prefix='pd_I')
# dataeq2=se.SweepData(folder13, prefix='pd_I')
# dataeq3=se.SweepData(folder14, prefix='pd_I')
# dataeq4=se.SweepData(folder15, prefix='pd_I')
# spatialeq1 = dataeq1.get_spatial_data(dataeq1.v_row(0))
# IB_maskeq1 = dataeq1.IB_mask(spatialeq1)  
# spatialeq2 = dataeq2.get_spatial_data(dataeq2.v_row(0))
# IB_maskeq2 = dataeq2.IB_mask(spatialeq2) 
# spatialeq3 = dataeq3.get_spatial_data(dataeq3.v_row(0))
# IB_maskeq3 = dataeq3.IB_mask(spatialeq3)   
# spatialeq4 = dataeq4.get_spatial_data(dataeq4.v_row(0))
# IB_maskeq4 = dataeq4.IB_mask(spatialeq4)

# plt.figure(4)
# plt.plot(spatial12['coord_x'][IB_mask12], spatial12['u_IB'][IB_mask12]/(1e17),color = '#00aa00', label='$\mu_I$ = 0.001') 
# plt.plot(spatial13['coord_x'][IB_mask13], spatial13['u_IB'][IB_mask13]/(1e17), color = '#0011ff', label='$\mu_I$ = 1') 
# plt.plot(spatial14['coord_x'][IB_mask14], spatial14['u_IB'][IB_mask14]/(1e17), color = '#F5B326', label='$\mu_I$ = 30') 
# plt.plot(spatial15['coord_x'][IB_mask15], spatial15['u_IB'][IB_mask15]/(1e17), color = '#F526BC', label='$\mu_I$ = 100') 
# plt.plot(spatialeq1['coord_x'][IB_maskeq1], spatialeq1['u_IB'][IB_maskeq1]/(1e17), linestyle='dashed', color = '#00aa00', label='equilibrium') 
# plt.plot(spatialeq2['coord_x'][IB_maskeq2], spatialeq2['u_IB'][IB_maskeq2]/(1e17), linestyle='dashed', color = '#0011ff') 
# plt.plot(spatialeq3['coord_x'][IB_maskeq3], spatialeq3['u_IB'][IB_maskeq3]/(1e17), linestyle='dashed', color = '#F5B326') 
# plt.plot(spatialeq4['coord_x'][IB_maskeq4], spatialeq4['u_IB'][IB_maskeq4]/(1e17), linestyle='dashed', color = '#F526BC') 
# plt.title("f_I vs position for GCM = 50 at mpp, $\sigma_{ci}$ = $\sigma_{iv}$ = 1e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_GCM=50__sigmaCI=sigmaIV=1e-13_with_equilibrium.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plt.figure(3)
# plt.plot(spatial8['coord_x'][IB_mask8], spatial8['u_IB'][IB_mask8]/(1e17), color = '#00aa00', linestyle = 'dashed', label='$m_G$ = 0%') 
# plt.plot(spatial9['coord_x'][IB_mask9], spatial9['u_IB'][IB_mask9]/(1e17), color = '#0011ff', linestyle = 'dashed', label='$m_G$ = 10%') 
# plt.plot(spatial10['coord_x'][IB_mask10], spatial10['u_IB'][IB_mask10]/(1e17), color = '#F5B326', linestyle = 'dashed', label='$m_G$ = 25%') 
# plt.plot(spatial11['coord_x'][IB_mask11], spatial11['u_IB'][IB_mask11]/(1e17), color = '#F526BC', linestyle = 'dashed', label='$m_G$ = 50%') 
# plt.title("f_I vs position for mu_I = 30 at mpp, sigma_ci = sigma_iv = 5e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_VS_position_mu=30_sigma_CI=sigma_IV=5e-13.png")

# plt.figure(4)
# plt.plot(spatial12['coord_x'][IB_mask12], spatial12['u_IB'][IB_mask12]/(1e17), color = '#00aa00', linestyle = 'dashed', label='$m_G$ = 0%') 
# plt.plot(spatial13['coord_x'][IB_mask13], spatial13['u_IB'][IB_mask13]/(1e17), color = '#0011ff', linestyle = 'dashed', label='$m_G$ = 10%') 
# plt.plot(spatial14['coord_x'][IB_mask14], spatial14['u_IB'][IB_mask14]/(1e17), color = '#F5B326', linestyle = 'dashed', label='$m_G$ = 25%') 
# plt.plot(spatial15['coord_x'][IB_mask15], spatial15['u_IB'][IB_mask15]/(1e17), color = '#F526BC', linestyle = 'dashed', label='$m_G$ = 50%') 
# plt.title("f_I vs position for mu_I = 100 at mpp, sigma_ci = sigma_iv = 5e-13", fontsize = 10) 
# plt.xlabel(r'x ($\mu$m)'), plt.ylabel(r'$f_I$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("MPP_f_I_VS_position_mu=100_sigma_CI=sigma_IV=5e-13.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #plotting JV curves   
# jv_plot(data0.jv)
# jv_plot(data1.jv)
# jv_plot(data2.jv)
# jv_plot(data3.jv)
# plt.legend([r'$\mu_I$ = 0.001', r'$\mu_I$ = 1', r'$\mu_I$ = 30', r'$\mu_I$ = 100'])
# plt.title(r"JV curves at different IB mobilities, GCM=0, $\sigma_{ci}$=$\sigma_{iv}$=5e-13")
# plt.savefig("JV_GCM=0__sigmaCI=sigmaIV=5e-13.png")
# plt.clf()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subgap_generation_mismatch_diagram(spatial0, IB_mask0)
plt.title("Optical Generation at mpp, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial1, IB_mask1)
plt.title("Optical Generation at mpp, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial2, IB_mask2)
plt.title("Optical Generation at mpp, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial3, IB_mask3)
plt.title("Optical Generation at mpp, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial4, IB_mask4)
plt.title("Optical Generation at mpp, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial5, IB_mask5)
plt.title("Optical Generation at mpp, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial6, IB_mask6)
plt.title("Optical Generation at mpp, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial7, IB_mask7)
plt.title("Optical Generation at mpp, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial8, IB_mask8)
plt.title("Optical Generation at mpp, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial9, IB_mask9)
plt.title("Optical Generation at mpp, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial10, IB_mask10)
plt.title("Optical Generation at mpp, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial11, IB_mask11)
plt.title("Optical Generation at mpp, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial12, IB_mask12)
plt.title("Optical Generation at mpp, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial13, IB_mask13)
plt.title("Optical Generation at mpp, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial14, IB_mask14)
plt.title("Optical Generation at mpp, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial15, IB_mask15)
plt.title("Optical Generation at mpp, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("MPP_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

# extract spatial data at SC
spatial0 = data0.get_spatial_data(data0.v_row(0.0))
IB_mask0 = data0.IB_mask(spatial0)  

spatial1 = data1.get_spatial_data(data1.v_row(0.0))
IB_mask1 = data1.IB_mask(spatial1)  

spatial2 = data2.get_spatial_data(data2.v_row(0.0))
IB_mask2 = data2.IB_mask(spatial2)  

spatial3 = data3.get_spatial_data(data3.v_row(0.0))
IB_mask3 = data3.IB_mask(spatial3)  

spatial4 = data4.get_spatial_data(data4.v_row(0.0))
IB_mask4 = data4.IB_mask(spatial4) 

spatial5 = data5.get_spatial_data(data5.v_row(0.0))
IB_mask5 = data5.IB_mask(spatial5)

spatial6 = data6.get_spatial_data(data6.v_row(0.0))
IB_mask6 = data6.IB_mask(spatial6)

spatial7 = data7.get_spatial_data(data7.v_row(0.0))
IB_mask7 = data7.IB_mask(spatial7)

spatial8 = data8.get_spatial_data(data8.v_row(0.0))
IB_mask8 = data8.IB_mask(spatial8)

spatial9 = data9.get_spatial_data(data9.v_row(0.0))
IB_mask9 = data9.IB_mask(spatial9)

spatial10 = data10.get_spatial_data(data10.v_row(0.0))
IB_mask10 = data10.IB_mask(spatial10)

spatial11 = data11.get_spatial_data(data11.v_row(0.0))
IB_mask11 = data11.IB_mask(spatial11)

spatial12 = data12.get_spatial_data(data12.v_row(0.0))
IB_mask12 = data12.IB_mask(spatial12)

spatial13 = data13.get_spatial_data(data13.v_row(0.0))
IB_mask13 = data13.IB_mask(spatial13)

spatial14 = data14.get_spatial_data(data14.v_row(0.0))
IB_mask14 = data14.IB_mask(spatial14)

spatial15 = data15.get_spatial_data(data15.v_row(0.0))
IB_mask15 = data15.IB_mask(spatial15)


subgap_generation_mismatch_diagram(spatial0, IB_mask0)
plt.title("Optical Generation at SC, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial1, IB_mask1)
plt.title("Optical Generation at SC, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial2, IB_mask2)
plt.title("Optical Generation at SC, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial3, IB_mask3)
plt.title("Optical Generation at SC, GCM=0, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=0__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial4, IB_mask4)
plt.title("Optical Generation at SC, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial5, IB_mask5)
plt.title("Optical Generation at SC, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial6, IB_mask6)
plt.title("Optical Generation at SC, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial7, IB_mask7)
plt.title("Optical Generation at SC, GCM=10, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=10__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial8, IB_mask8)
plt.title("Optical Generation at SC, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial9, IB_mask9)
plt.title("Optical Generation at SC, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial10, IB_mask10)
plt.title("Optical Generation at SC, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial11, IB_mask11)
plt.title("Optical Generation at SC, GCM=25, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=25__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial12, IB_mask12)
plt.title("Optical Generation at SC, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 0.001")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=0.001.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial13, IB_mask13)
plt.title("Optical Generation at SC, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 1")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=1.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial14, IB_mask14)
plt.title("Optical Generation at SC, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 30")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=30.png")
plt.clf()

subgap_generation_mismatch_diagram(spatial15, IB_mask15)
plt.title("Optical Generation at SC, GCM=50, $\sigma_{ci}$=5*$\sigma_{iv}$, $\mu_I$ = 100")
plt.tight_layout()
plt.savefig("SC_OptGen_GCM=50__sigmaCI=5sigmaIV_mu=100.png")
plt.clf()
