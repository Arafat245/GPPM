import os
import matplotlib
## workaround for OS X
from sys import platform as sys_pf
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import GP_progression_model
import torch
import pandas as pd
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Input to the script
#  - csv path (e.g. "./table_APOEposRID.csv")
#  - list of fields (e.g. ["Hippocampus","Ventricles","Entorhinal","WholeBrain","ADAS11","FAQ","AV45","FDG"]  )  
#  - list of {-1,0,1} for monotonicity: [-1,1,-1,-1,1,1,1,-1]
#  - int for monotonicity trade_off: 100

# Import data #
pseudo_adni = pd.read_csv('http://marcolorenzi.github.io/material/pseudo_adni.csv')

print(pseudo_adni.columns)

diags = { 'NL' : np.sum(pseudo_adni['group']==1), 
          'MCI' : np.sum(pseudo_adni['group']==2), 
          'AD' : np.sum(pseudo_adni['group']==3), }

biomarkers = ['Hippocampus', 'Ventricles', 'Entorhinal', 'WholeBrain',
       'ADAS11', 'FAQ', 'AV45', 'FDG']
Xdata, Ydata, RID, list_biomarkers, group = GP_progression_model.convert_from_df(pseudo_adni,
                                                                 biomarkers, time_var = 'Time')

# Initialize the model
dict_monotonicity = { 'Hippocampus' : -1, #decreasing
                     'Ventricles': 1, #increasing
                     'Entorhinal': -1, #decreasing
                     'WholeBrain': -1, #decreasing
                     'ADAS11': 1, #increasing
                     'FAQ': 1, #increasing
                     'AV45': 1, #increasing
                     'FDG': -1, #decreasing
                    }

# trade off between data-fit and monotonicity
trade_off = 100

# create a GPPM object
model = GP_progression_model.GP_Progression_Model(Xdata,Ydata, names_biomarkers = biomarkers, monotonicity = [dict_monotonicity[k] for k in dict_monotonicity.keys()], trade_off = trade_off,
                                                  groups = group, group_names = ['NL','MCI','AD'], device = device)
model.model = model.model.to(device)

# Optimise the model
N_outer_iterations = 6
N_iterations = 200
model.Optimize(N_outer_iterations = N_outer_iterations, N_iterations = N_iterations, n_minibatch = 10, verbose = True, plot = False, benchmark = True)

# Visulization of the results
model.Plot()

model.Plot(save_fig = './')
model.Plot(save_fig = './', joint = True)

plt.figure(figsize=(6,6))
timing = plt.imread('./change_timing.png')
plt.imshow(timing, aspect='auto')
plt.tight_layout()
plt.title('Estimated trajectories timing')


plt.figure(figsize=(8,8))
timing = plt.imread('./change_magnitude.png')
plt.imshow(timing, aspect='auto')
plt.tight_layout()
plt.title('Estimated trajectories magnitude')

# Prediction of the disease severity per subject
pred_time_shift = model.PredictOptimumTime(pseudo_adni, id_var='RID', time_var='Time')

# This command creates another .csv file
pred_time_shift.to_csv('./predictions.csv')

diag_dict = dict([(1,'NL'),(2,'MCI'),(3,'AD')])
pred_time_shift.reset_index(inplace=True)

rid_group = pseudo_adni.loc[~pseudo_adni['RID'].duplicated(),][['RID','group']]
group_and_ts = rid_group.merge(pred_time_shift, on = 'RID')

group_and_ts = rid_group.merge(pred_time_shift, on = 'RID')

fig, ax = plt.subplots(figsize=(8,6))
for label, df in group_and_ts.groupby('group'):
    if len(df['Time Shift'])>1:
        df['Time Shift'].plot(kind="kde", ax=ax, label= diag_dict[label])
    else:
        print('Warning: ' + label + ' group has only 1 element and will not be displayed')
plt.legend()

