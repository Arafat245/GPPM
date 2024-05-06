import numpy as np
import matplotlib
## workaround for OS X
from sys import platform as sys_pf
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import GP_progression_model
from GP_progression_model import DataGenerator
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Parameters for progression curves and data generation
# Max value of the trajectories 
# (we assume that they range from 0, healthy, to L, pathological)
L = 1
# time interval for the trajectories
interval = [-15,15]

# Number of biomarkers
Nbiom = 4
# Number of individuals
Nsubs = 30
# Gaussian observational noise
noise = 0.1

# We set the seed=111 as a reference for the further analysis
seed = 111
np.random.seed(seed)

# Creating random parameter sets for each biomarker's progression
flag = 0
while (flag!=1):
    CurveParam = []
    for i in range(Nbiom):
        CurveParam.append([L,1*(0.5+np.random.rand()),noise])
        if CurveParam[i][1] > 0.0:
            flag = 1


# Calling the data generator
dg = DataGenerator.DataGenerator(Nbiom, interval, CurveParam, Nsubs, seed)

dg.plot('long')

# The time_shift indicates the moment at which each biomarker becomes abnormal
ground_truth_ts = dg.time_shift
dict_gt_ts = {}
for biom in range(Nbiom):
  dict_gt_ts['biom_'+ str(biom)] = ground_truth_ts[biom]

pd.DataFrame(dict_gt_ts, index=['change time'])

dg.plot('short')
plt.show()

df  = dg.get_df()
# since the biomarkers are increasing from normal to abnormal states 
# (resp. from 0 to L) we set the monotonicity constraint to 1 for each biomarker
monotonicity = np.repeat(1,Nbiom)

# the input is the generated synthetic data frame
input_data = df.drop(columns=['time_gt'])

# we convert the input data to the pieces that will be fed to GPPM  
Xdata, Ydata, RID, list_biomarkers, group = GP_progression_model.convert_from_df(input_data,
                                                                 ['biom_' + str(i) for i in range(Nbiom)], time_var = 'time')

# here we call GPPM with the appropriate input
model = GP_progression_model.GP_Progression_Model(Xdata,Ydata, monotonicity = monotonicity, trade_off = 50,
                                                  names_biomarkers=['biom_' + str(i) for i in range(Nbiom)] )

model.Optimize(plot = True, verbose = True, benchmark=True)

#This command saves the fitted biomarkers trajectories in separate files
model.Plot(save_fig = './')

plt.figure(figsize=(10,10))
for i in range(Nbiom):
  plt.subplot(2,2,i+1)
  fig_biom = plt.imread('./biom_'+str(i)+'.png')
  plt.imshow(fig_biom)
plt.show()

# with the option 
model.Plot(save_fig = './', joint = True)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
timing = plt.imread('./change_timing.png')
plt.imshow(timing, aspect='auto')
plt.title('Estimated trajectories timing')
plt.subplot(1,2,2)
dg.plot('long')
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
magnitude = plt.imread('./change_magnitude.png')
plt.imshow(magnitude, aspect = 'auto')
plt.title('Estimated trajectories magnitude')
plt.subplot(1,2,2)
dg.plot('long')
plt.show()

# predicting individual disease severity
pred_time_shift = model.PredictOptimumTime(input_data, id_var='RID', time_var='time')

# plotting predicted vs ground truth
ground_truth = dg.OutputTimeShift()

plt.scatter(pred_time_shift['Time Shift'],ground_truth) 
plt.xlabel('estimated time-shift')
plt.ylabel('ground truth disease stage')
plt.show()




