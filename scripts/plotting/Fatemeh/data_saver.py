
#modified version of 'open_optimizer_summary.py' (modified by Fatemeh.MK)
from simudo.io import h5yaml
from glob import glob
from pathlib import Path
import csv

#for running within directory containing data.yaml file
from simudo.io import h5yaml
with open('data.yaml') as file:
    summary = h5yaml.load(file)
print(summary) 

#saves all of the parameters and optimization results in a csv file
with open('data.csv', 'w') as f:
   for key, value in summary.items():
      if key == "params":
         for param, value in summary['params'].items():
            csv.writer(f).writerow([param, value])
            if param == 'multiplex_keys':
               if value == None :
                  multiplex = False
               else:
                  multiplex = True
      else:
         csv.writer(f).writerow([key, value])

print(multiplex)
if multiplex == True:
   #for running within directory containing all run folders when multiplexing
   files = glob("*/optimizer_summary.yaml")
   summaries = []
   for f in files:
      folders = [f.split("/")[0]] #splits string folder/optimizer_summary.yaml and only takes folder number 
      with open(f) as curf:
         summary = h5yaml.load(curf)
         summaries.append(summary)
         #print(f, summary)
         submitfile = folders[0] + "/submit.yaml"
         #submitfile = glob(int(folders[0]),"/submit.yaml")
         #print(submitfile) 
         with open(submitfile) as curf:
            params = h5yaml.load(curf) 
            IB_E = params['parameters']['IB_E']
            X = params['parameters']['X']
            mu_I = params['parameters']['mu_I']
            sigma_opt_ci = params['parameters']['sigma_opt_ci']
            sigma_opt_iv = params['parameters']['sigma_opt_iv']
            optimize_key = params['parameters']['optimize_key']
            #print(", IB_E = ", IB_E, ", X = ", X, ", mu_I = ", mu_I, ", sigma_opt_ci = ", sigma_opt_ci, ", sigma_opt_iv = ", sigma_opt_iv)
      print("Folder", folders[0], summary, " IB_E =", IB_E, " X =", X, " mu_I =", mu_I, " sigma_opt_ci =", sigma_opt_ci, " sigma_opt_iv =", sigma_opt_iv)
else:
   #for running within directory containing all run folders when NOT multiplexing
   files = glob("optimizer_summary.yaml")
   summaries = []
   for f in files:
      with open(f) as curf:
         summary = h5yaml.load(curf)
         summaries.append(summary)
         #print(f, summary)
         submitfile =  "submit.yaml"
         #submitfile = glob(int(folders[0]),"/submit.yaml")
         #print(submitfile) 
         with open(submitfile) as curf:
            params = h5yaml.load(curf) 
            IB_E = params['parameters']['IB_E']
            X = params['parameters']['X']
            mu_I = params['parameters']['mu_I']
            sigma_opt_ci = params['parameters']['sigma_opt_ci']
            sigma_opt_iv = params['parameters']['sigma_opt_iv']
            optimize_key = params['parameters']['optimize_key']

#saves the optimization summaries in a text file
with open('optimization_summary.txt', 'w') as f:
   f.write("Optimized for efficiency over " + optimize_key +'.\n')
   ctr = 1 #counter
   for sum in summaries:
      f.write(f'Run {ctr:.0f}:' + str(sum) + '\n')
      ctr+=1

#saves all of the parameters in a csv file
param_dict = params['parameters']
param_list = sorted(param_dict.items()) 
with open('parameters.csv', 'w') as f:
   for pair in param_list:
      csv.writer(f).writerow([pair[0], pair[1]])