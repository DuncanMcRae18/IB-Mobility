# Intermediate Band mobility research
### Duncan McRae

This folder contains the scripts, plots and a limited amout of data which should allow anyone to recreate the results I have found myself. To recreate the environment you'll need to use miniconda to create the duncan_env_full.yml to run the Simudo software. Then you can run Four_layer_IB_Eg_1.67.py and/or Four_layer_IB_Eg_2.5.py (adjust the output path) to recreate my data. The project is divided into six main directories:
```
├── data  
├── docs  
├── envs  
├── fossil  
├── plots  
└── scripts  
```
## data
I included data at the short-citcuit, open-circuit and max-power voltages to avoid making the repo to large. This should still be adequate to do most plotting as needed. To fully recreate the data, the scripts in scripts/data_creation can be used to simulate the same problem. The database with data such as efficiency, thickness, etc. is also in here called duncan_results.db.
The data is collected into two folders:  
**duncan**  
Contains all of the data collected by me, organized as follows: Eg, sigma_opt_ci, sigma_opt_iv, m_G value (global current match).  
```
├── Eg_1.67  
│   ├── mu_I_0.001  
│   │   ├── ci_1.0e-13  
│   │   │   ├── iv_1.0e-13  
│   │   │   │   ├── GCM_0  
│   │   │   │   ├── GCM_10  
│   │   │   │   ├── GCM_25  
│   │   │   │   └── GCM_50  
│   │   │   └── iv_5.0e-13  
│   │   │       ├── GCM_0  
│   │   │       ├── GCM_10  
│   │   │       ├── GCM_25  
│   │   │       └── GCM_50  
│   │   └── ci_5.0e-13  
│   │   ...  
└── Eg_2.5  
    ├── mu_I_0.001  
    │   ├── ci_1.0e-13  
    │   │   ├── iv_1.0e-13  
    │   │   │   ├── GCM_0  
    │   │   │   ├── GCM_10  
    │   │   │   ├── GCM_25  
    │   │   │   └── GCM_50  
    │   │   └── iv_5.0e-13  
    │   │       ├── GCM_0  
    │   │       ├── GCM_10  
    │   │       ├── GCM_25  
    │   │       └── GCM_50  
    │   └── ci_5.0e-13  
│   │   ...  
```
**fatemeh**  
Contains Fatemehs data organized by date.  
## docs
All of my sources, reference data, reports, etc.  
## envs
Contain the python environment details, created from both my environment (duncan_env_full.py) and the approximate environment of a previous researcher Fatemeh Mousavi  (fatemeh_env_full.py). Her environment is preserverd primarily caused by an issue in the package versions.  
## fossil
A fossil repository containing my version of the Simudo software, with minor adjustments made to the maximum PDD solver iterations in simudo/physics/steppers.py in order to ensure the Newton solver converged.  
## plot  
Scripts in scripts/plotting will save plots here.  
## scripts  
The main location of my scripts to simulate, organize and analyze data. Contains five main directories:  
&emsp;**database**  
&emsp;&emsp;Scripts to create and edit the database file duncan_results.db. The script used to create fatemeh_results.db was deleted.  
&emsp;**plotting**  
&emsp;&emsp;Scripts used to plot from the data within this repo. You need to change the file paths.  
&emsp;**random**  
&emsp;&emsp;Scripts for medial things and to demonstrate Fatemeh's version issue.  
&emsp;**simulate**  
&emsp;&emsp;Files to create simulation data. Change the file paths first.  
&emsp;**warmup**  
&emsp;&emsp;python scripts from beginning stages.  