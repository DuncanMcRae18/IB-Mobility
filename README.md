# Intermediate Band mobility research
### Duncan McRae

## Intro
This folder contains the scripts, plots and a limited amout of data which should allow any user to recreate the results I have found myself. The project is divided into six main directories:

├── data  
├── docs  
├── envs  
├── fossil  
├── plots  
└── scripts  

## data
I included data at the short-citcuit, open-circuit and max-power voltages to avoid making the repo to large. To fully recreate the data, the scripts in scripts/data_creation can be used to simulate the same problem.  
## docs
All of my sources, reference data, etc.  
## envs
Contain the python environment details, created from both my environment and the approximate environment of a previous researcher Fatemeh Mousavi Karimi. Her environment is preserverd primarily caused by an issue in the package versions.  
## fossil
A fossil repository containing my version of the Simudo software, with minor adjustments made to parameters in order to ensure the Newton solver converged.  
## plot 
Scripts in scripts/plotting will save plots here.
## scripts
The main location of my scripts to simulate, organize and analyze data. Contains three main directories:
