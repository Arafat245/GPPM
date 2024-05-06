import matplotlib
## workaround for OS X
from sys import platform as sys_pf
if sys_pf == 'darwin':
    matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.optim
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy
import pandas as pd
import pickle 
import seaborn as sns
import torch._utils
import time
import math

class GP_Progression_Model(object):
    def  __init__(self, x = [], y = [], monotonicity = [], trade_off = 0, reparameterization_model = 'time_shift', groups = [],
                  group_names = [] ,names_biomarkers = [], device = 'cpu'):
        self.x = deepcopy(x)
        self.y = deepcopy(y) 
        self.N_biomarkers = len(x)

        if self.N_biomarkers>0:
            self.N_subs = len(x[0])
        else:
            self.N_subs = 0

        self.names_biomarkers = names_biomarkers
        self.groups = groups
        self.group_name = group_names

        self.reparameterization_model = reparameterization_model
        self.monotonicity = monotonicity
        self.trade_off = trade_off

        self.device = device

        self.x_mean_std = []
        self.y_mean_std = []

        self.x_torch = []
        self.y_torch = []

        if self.N_biomarkers>0:
            self.Initialize(self.device)

        self.tr = Time_reparameterization(self.N_subs, self.N_biomarkers, self.reparameterization_model, self.device)
        self.model = Regression_Model(self.tr, self.N_subs, self.N_biomarkers, self.device)

    def Initialize(self, first = True):

        x_data = np.hstack([np.hstack(self.x[i]) for i in range(self.N_biomarkers)])
        x_mean = np.mean(x_data)
        x_std = np.std(x_data)
        if first:
            for b in range(self.N_biomarkers):
                self.x_mean_std.append([x_mean,x_std])
                self.y_mean_std.append([np.mean(np.hstack(self.y[b])),np.std(np.hstack(self.y[b]))])
                for sub in range(len(self.x[b])):
                    self.x[b][sub] = (self.x[b][sub]-self.x_mean_std[b][0])/self.x_mean_std[b][1]
                    self.y[b][sub] = (self.y[b][sub]-self.y_mean_std[b][0])/self.y_mean_std[b][1]

        for b in range(len(self.x)):
            self.x_torch.append([torch.Tensor(i).to(self.device) for i in self.x[b]])
            self.y_torch.append(torch.cat([Variable(torch.Tensor(i)).to(self.device) for i in self.y[b]]))

    def Optimize(self, N_outer_iterations = 6, N_iterations = 200, n_minibatch = 1, verbose = False, plot = False, benchmark = False):
        time_optimizer = torch.optim.Adam([{'params': self.model.branches[0][0].parameters(), 'lr': 1e-2}])
        regression_parameters = [{'params': self.model.branches[i][1].parameters(), 'lr': 1e-2} for i in range(self.N_biomarkers)]       
        regression_parameters.append({'params': self.model.branches[self.N_biomarkers].parameters(), 'lr': 1e-2}) 
        regression_parameters.append({'params':self.model.branches[self.N_biomarkers+1].parameters(), 'lr':1e-2})
        GP_optimizer = torch.optim.Adam(regression_parameters)

        self.tr.list_id = np.sort(np.random.choice(len(self.x[0]), round(self.N_subs), replace=False)).tolist()
        if not len(self.monotonicity)==self.N_biomarkers:
            self.monotonicity = []
            for i in range(self.N_biomarkers):
                self.monotonicity.append(1)

        x_min = float('inf')
        x_max = float('-inf')
        for bio_pos,biom in enumerate(range(self.N_biomarkers)):
            if self.reparameterization_model == 'time_shift':
                x_data = ((self.model.time_reparameterization(self.x_torch)[biom]).detach().data.cpu().numpy())
            else:
                x_data = ((self.model.time_reparameterization(self.x_torch)[biom]).detach().data.cpu().numpy()[:, 0])
            if (float(np.min(x_data)) < x_min):
                x_min = float(np.min(x_data))
            if (float(np.max(x_data)) > x_max):
                x_max = float(np.max(x_data))

        delta = (x_max-x_min)/5
        x_range = Variable(torch.arange(x_min-delta,x_max + delta, float((x_max + 2*delta - x_min)/(2.0*len(self.x[0]))))).to(self.device)
        x_range = x_range.reshape(x_range.size()[0],1)

        ## constraints
        N_obs_tot = torch.cat([torch.cat(self.x_torch[i]) for i in range(self.N_biomarkers)]).size()[0]
        x_constraint = []
        for biom in range(self.N_biomarkers):
            x_constraint.append([])
            x_constraint[biom] =  x_range #.repeat(int(N_obs_tot/(len(x_range)*self.N_biomarkers)),1)  

        for j in range(N_outer_iterations):
            if verbose:
                print('Optimization step: ' + str(j+1) + ' out of ' + str(N_outer_iterations))
                print (' -- Regression --')
#            for biom in range(self.N_biomarkers):
#                self.model.branches[biom][1].Reset()
            ## mini-batch implementation
            # for each Outer_iteration, re-subsample the subject space in n_minibatch number of minibatches
            if j == N_outer_iterations - 2:  # last step of Regression + Time_rep (+ Regression) is always full batch
                n_minibatch = 1
            minibatches = partition(self.N_subs,n_minibatch)
            # print(minibatches)

            for i in range(N_iterations):

                # for each iteration, use a different minibatch from the set of minibatches.
                list_id = np.sort(minibatches[i % n_minibatch])
                self.model.branches[self.N_biomarkers + 1].list_id = list_id
                self.tr.list_id = list_id
                self.model.time_reparameterization.list_id = list_id

                x_torch_mb, y_torch_mb = self.Create_minibatch(list_id = list_id)

                ## Optimization
                tic = time.perf_counter()

                self.model.zero_grad()
                pred = self.model(x_torch_mb)
                pred_constraint = []
                for biom in range(self.N_biomarkers):
                    pred_constraint.append(self.model.branches[biom][1](x_constraint[biom]))

                cost_ll = -Cost(y_torch_mb, pred, pred_constraint, self.monotonicity, self.trade_off, self.device)[0]
                cost_constr = -Cost(y_torch_mb, pred, pred_constraint, self.monotonicity, self.trade_off, self.device)[1]
                # cost = -Cost(y_torch_mb, pred, pred_constraint, self.trade_off, self.device)
                L = cost_ll + cost_constr + self.model.KL()
                L.backward()
                GP_optimizer.step()

                toc = time.perf_counter()

                if verbose and ((i==0) or (i+1)%50==0):
                    print('Iteration ' + str(i+1) + ' of ' + str(N_iterations) + ' || Cost (DKL): %.2f' % self.model.KL().item() +
                          ' - Cost (fit): %.2f' % cost_ll.item() + ' - Cost (constr): %.2f' % cost_constr.item() +
                          '|| Batch (each iter) of size %d' % len(list_id), end =" ")
                    if benchmark:
                        print('|| Time (each iter): %.2f' % (toc - tic) + 's')
            if plot:
                self.Plot()

            if j<N_outer_iterations-1:
                if verbose:
                    print (' -- Time reparameterization --')

                for i in range(N_iterations):

                    # for each iteration, use a different minibatch from the set of minibatches.
                    list_id = np.sort(minibatches[i % n_minibatch])
                    self.model.branches[self.N_biomarkers + 1].list_id = list_id
                    self.tr.list_id = list_id
                    self.model.time_reparameterization.list_id = list_id

                    x_torch_mb, y_torch_mb = self.Create_minibatch(list_id=list_id)

                    tic = time.perf_counter()

                    self.model.zero_grad()
                    pred = self.model(x_torch_mb)
                    pred_constraint = []
                    for biom in range(self.N_biomarkers):
                        pred_constraint.append(self.model.branches[biom][1](x_constraint[biom]))

                    cost_ll = -Cost(y_torch_mb, pred, pred_constraint, self.monotonicity, self.trade_off, self.device)[0]
                    cost_constr = -Cost(y_torch_mb, pred, pred_constraint, self.monotonicity, self.trade_off, self.device)[1]
                    # cost = -Cost(y_torch_mb, pred, pred_constraint, self.trade_off, self.device)
                    L = cost_ll + cost_constr + self.model.KL()
                    L.backward()
                    time_optimizer.step()

                    toc = time.perf_counter()

                    if verbose and ((i==0) or (i+1)%50==0):
                        print('Iteration ' + str(i + 1) + ' of ' + str(N_iterations) + ' || Cost (DKL): %.2f' % self.model.KL().item() +
                              ' - Cost (fit): %.2f' % cost_ll.item() + ' - Cost (constr): %.2f' % cost_constr.item()+
                          '|| Batch (each iter) of size %d' % len(list_id), end =" ")
                        if benchmark:
                            print('|| Time (each iter): %.2f' % (toc - tic) + 's')

                if plot:
                    self.Plot()

    def Create_minibatch(self, list_id = []):
        x_torch_mb = []
        y_torch_mb = []
        y_mb = []

        for b in range(len(self.x)):
            x_torch_mb.append([])
            y_mb.append([])
            for s in list_id:
                x_torch_mb[b].append(self.x_torch[b][s])
                y_mb[b].append(self.y[b][s])
        for b in range(len(self.x)):
            y_torch_mb.append(torch.cat([Variable(torch.Tensor(i)).to(self.device) for i in y_mb[b]]))

        return x_torch_mb, y_torch_mb

    def ReturnTimeParameters(self, save_fig = ''):
        if self.reparameterization_model == 'time_shift':
            tp = self.tr.time_parameters.detach().data.cpu().numpy() *  self.x_mean_std[0][1] + self.x_mean_std[0][0]

            if save_fig:
                p_df = pd.DataFrame({"class": self.groups, "vals": tp})
                fig, ax = plt.subplots(figsize=(8,6))
                for label, df in p_df.groupby('class'):
                    if len(df.vals)>1:
                        df.vals.plot(kind="kde", ax=ax, label= label)
                    else:
                        print('Warning: ' + str(label) + ' group has only 1 element and will not be displayed')
                plt.legend()
                plt.savefig(save_fig + '/time_shift_distribution.png' )

            return tp
        else:
            return self.tr.time_parameters0.detach().data.cpu().numpy(), self.tr.time_parameters1.detach().numpy()


    def Plot(self, list_biom = [], joint = False, save_fig = ''):
        if len(list_biom)<1:
            list_biom = self.names_biomarkers
        self.tr.list_id = np.arange(len(self.x[0])).tolist()
        x_min = float('inf')
        x_max = float('-inf')

        y_min = float('inf')
        y_max = float('-inf')

        for bio_pos,biomarker in enumerate(list_biom):
            bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
            x_data = (self.model.time_reparameterization(self.x_torch)[bio_id]).detach().data.cpu().numpy()
            y_data = (self.y_torch[bio_id]).detach().data.cpu().numpy()
            if (float(np.min(x_data)) < x_min):
                x_min = float(np.min(x_data))
            if (float(np.max(x_data)) > x_max):
                x_max = float(np.max(x_data))
            
            if (float(np.min(y_data)) < y_min):
                y_min = float(np.min(y_data))
            if (float(np.max(y_data)) > y_max):
                y_max = float(np.max(y_data))

        x_range = Variable(torch.arange(x_min,x_max, float((x_max-x_min)/50)))
        x_range = x_range.reshape(x_range.size()[0],1)

        new_x = self.Transform_subjects()


        if not joint:
            fig = plt.figure()
            for bio_pos,biomarker in enumerate(list_biom):
                bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
                predictions = []
                for i in range(200):
                    predictions.append(self.model.branches[bio_id][1](x_range))

                if not save_fig:
                    if len(list_biom)>3:
                        ax = fig.add_subplot(math.ceil((len(list_biom) - 1) / 3) + 1, 3, (bio_pos) + 1)
                    else:
                        ax = fig.add_subplot(1, len(list_biom) , (bio_pos) + 1)
                else:
                    ax = fig.add_subplot(1, 1,  1)
                if len(self.names_biomarkers)>0:
                    plt.title(self.names_biomarkers[bio_id])

                ax.set_xlim(x_min * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], x_max * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0])
                ax.set_ylim(y_min, y_max)
                mean = np.mean([predictions[i][:, 0].data.cpu().numpy() for i in range(200)], axis=0)

                for i in range(200):
                    ax.plot(x_range.numpy() * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], predictions[i][:, 0].data.cpu().numpy(), lw=0.05, color='red')
                ax.plot( x_range.numpy() * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], mean, lw=2, color='black')


                for sub in range(self.N_subs):
                    x_data = new_x[bio_id][sub]
                    y_data = self.y[bio_id][sub]

                    if len(self.groups)>0:
                        palette = plt.get_cmap('tab10')
                        group_names = np.unique(self.groups)
                        group_sub  = np.where([group_names[i] == self.groups[sub] for i in range(len(group_names))])[0][0]
                        col = palette.colors[int(group_sub)]
                        if len(x_data)>1:
                            ax.plot( x_data * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], y_data, color=col, lw=1)
                        else:
                            ax.scatter( x_data * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], y_data, c=[col], s=1)
                    else:
                        col = 'green'
                        if len(x_data)>1:
                            ax.plot( x_data * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], y_data, color=col, lw=1)
                        else:
                            ax.scatter( x_data * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], y_data, c=[col], s=1)
                    if len(self.groups)>0:
                        patches = []
                        for index,i in enumerate(np.sort(np.unique(self.groups))):
                            palette = plt.get_cmap('tab10')
                            group_names = np.unique(self.groups)
                            group_sub  = np.where([group_names[i] == self.groups[sub] for i in range(len(group_names))])[0][0]
                            col = palette.colors[index]
                            patches.append(mpatches.Patch(color=col , label = i))
                            ax.legend(handles=patches)
                if save_fig:
                    plt.savefig(save_fig + biomarker + '.png')
                    plt.close()
                    fig = plt.figure()

            if not save_fig:
                plt.show()
            plt.close()

        else:
            fig = plt.figure(figsize=(100,50))
            ax = fig.add_subplot(1, 2 , 1)
            plt.title('Biomarkers', fontsize=100)
            patches = []
            if len(list_biom) <10:
                palette = plt.get_cmap('tab10')
            else:
                palette = plt.get_cmap('Spectral')

            for bio_pos,biomarker in enumerate(list_biom):
                bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
                predictions = []
                for i in range(200):
                    predictions.append(self.model.branches[bio_id][1](x_range))

                if len(list_biom) <10:
                    col = palette.colors[int(bio_pos)]
                else:
                    col = palette(int(float(bio_pos)/len(list_biom) * 256.0))

                mean = np.mean([predictions[i][:, 0].data.cpu().numpy() for i in range(200)], axis=0)
                ax.plot( x_range.numpy() * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], mean, color=col, lw=5)
                if len(self.names_biomarkers)>0:
                    patches.append(mpatches.Patch(color=col , label = biomarker))
                else:
                    patches.append(mpatches.Patch(color=col , label = 'Biomarker ' + str(biomarker)))
            ax.legend(handles=patches, fontsize=40)
            ax.tick_params(labelsize = 40)

            axD = fig.add_subplot(1, 2 , 2)
            plt.title('Derivatives', fontsize=100)
            for bio_pos,biomarker in enumerate(list_biom):
                predictions = []
                bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
                for i in range(200):
                    predictions.append(self.model.branches[bio_id][1](x_range))

                meanD = np.mean([predictions[i][:, 1].data.cpu().numpy() for i in range(200)], axis=0)
                if len(list_biom) <10:
                    col = palette.colors[int(bio_pos)]
                else:
                    col = palette(int(float(bio_pos)/len(list_biom) * 256.0))
                axD.plot( x_range.numpy() * self.x_mean_std[bio_id][1] + self.x_mean_std[bio_id][0], meanD, color=col, lw=5)
            if save_fig:
                axD.legend(handles=patches, fontsize=40)
                axD.tick_params(labelsize = 40)
                plt.savefig(save_fig + '/model_derivatives_plot.png' )
                plt.close()
            else:
                axD.legend(handles=patches, loc=2)
                plt.show()

            mean_sd = [] 
            list_times = []
            magnitude = []
            for bio_pos,biomarker in enumerate(self.names_biomarkers):
                preds = []
                for i in range(1000):
                    preds.append(self.model.branches[bio_pos][1](x_range))
                if (int(self.monotonicity[bio_pos])<0):
                    time_val = []
                    speed = []
                    for i in range(1000):
                        min_pos = np.argmin(preds[i][:, 1].data.cpu().numpy())
                        time_val.append(x_range.data.cpu().numpy()[min_pos] * self.x_mean_std[bio_pos][1] + self.x_mean_std[bio_pos][0])
                        speed.append(np.mean(preds[i][:, 1].data.cpu().numpy()))
                elif (int(self.monotonicity[bio_pos])>0):
                    time_val = []
                    speed = []
                    for i in range(200):
                        max_pos = np.argmax(preds[i][:,1].data.cpu().numpy())
                        time_val.append(x_range.data.cpu().numpy()[max_pos] * self.x_mean_std[bio_pos][1] + self.x_mean_std[bio_pos][0])
                        speed.append(np.mean(preds[i][:,1].data.cpu().numpy()))
                else:
                    time_val = 0

                mean_magnitudeD = np.mean(speed)
                sd_magnitude = np.std(speed)
                meanD = np.mean(time_val)
                stdD = np.std(time_val)
                list_times.append(np.array(time_val).flatten())
                mean_sd.append([meanD,stdD])
                magnitude.append([np.abs(mean_magnitudeD),sd_magnitude])

            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.tight_layout()
            order_mean = [np.mean(list_times[i]) for i in range(len(self.names_biomarkers))]
            order_biom = np.argsort(order_mean)
            df = pd.DataFrame([list_times[i] for i in order_biom], index=[self.names_biomarkers[i] for i in order_biom])
            df.T.boxplot(vert=False)
            plt.subplots_adjust(left=0.25)
            plt.title('Max change time')
            plt.xlabel('time')
            plt.savefig( save_fig + '/change_timing.png')
            plt.close()

            year_max_change = pd.DataFrame(np.vstack([[mean_sd[i][0] for i in order_biom],[mean_sd[i][1] for i in order_biom],[self.names_biomarkers[i] for i in order_biom]]).transpose())
            year_max_change.rename(index=str,columns={0:'mean time',1:'sd ',2:'biomarker'},inplace=True)
            year_max_change.to_csv( save_fig + '/max_change.csv')

            plt.figure(figsize=(10,15))
            order_magnitude = np.argsort([magnitude[i][0] for i in range(len(magnitude))])            
            plt.title('Average magnitude of change')
            plt.bar(range(len(magnitude)),[magnitude[i][0] for i in order_magnitude], yerr = [magnitude[i][1] for i in order_magnitude], alpha=0.2, align='center')
            plt.xticks(range(len(magnitude)), [self.names_biomarkers[i] for i in order_magnitude], rotation='vertical')
            plt.rcParams['ytick.labelsize'] = 'small'
            plt.savefig( save_fig + '/change_magnitude.png')
            plt.close()


    def Predict(self, x_test,y_test):
        N_sub_test = len(x_test[0])
        self.tr.list_id = np.sort(np.random.choice(len(self.x[0]), round(self.N_subs), replace=False)).tolist()
        x_min = float('inf')
        x_max = float('-inf')
        for bio_pos,biom in enumerate(range(self.N_biomarkers)):
            x_data = ((self.model.time_reparameterization(self.x_torch)[biom]).detach().data.cpu().numpy())
            if (float(np.min(x_data)) < x_min):
                x_min = float(np.min(x_data))
            if (float(np.max(x_data)) > x_max):
                x_max = float(np.max(x_data))

        x_range = Variable(torch.arange(x_min,x_max, float((x_max-x_min)/30)))
        x_range = x_range.reshape(x_range.size()[0],1)

        predicted_time = []
        for sub in range(N_sub_test):
            predicted_time.append([])
            print('Prediction for subject: ' + str(sub))
            for biom in range(self.N_biomarkers):
                predicted_time[sub].append([])
                for t_pos, time in enumerate(x_range):
                    time_prob = np.nan 
                    prob = 0
                    if len(x_test[biom][sub])>0:
                        x_test_sub = (x_test[biom][sub] - self.x_mean_std[biom][0])/self.x_mean_std[biom][1]
                        x_shifted = x_test_sub + time.numpy()
                        if x_shifted[0]<x_max: 
                            for i in range(100):
                                pred_test = self.model.branches[biom][1](Variable(torch.Tensor(x_shifted)).to(self.device).reshape(len(x_shifted),1))[:,0]
                                y_sub = (y_test[biom][sub] - self.y_mean_std[biom][0])/self.y_mean_std[biom][1]
                                prob += .5 * (pred_test.size()[0] * self.model.branches[self.N_biomarkers].sigma[biom].detach().data.cpu().numpy()
                                              + torch.sum((pred_test - Variable(torch.Tensor(y_sub)).to(self.device)) ** 2
                                                          / torch.exp(self.model.branches[self.N_biomarkers].sigma[biom].detach())
                                                          ).detach().data.cpu().numpy())
                                time_prob = prob/100
                    predicted_time[sub][biom].append(time_prob)
        return predicted_time

    def PredictOptimumTime(self, df_input, id_var='ID', time_var='Time'):
        """
        Receives a dataframe in which we have an ID column and a time column indicating the ID of the subject
        and the timepoint. If you df is indexed as [ID, TP], reset the index before-hand to have them as columns.
        Returns a dataframe containing the Time Shift for every ID and Time

        :param df_input: The input dataframe
        :param id_var: Variable in the columns of the dataframe that we will use as ID
        :param time_var: Variable in the columns of the dataframe to identify the timepoints
        :return: optim_time: Df containing the dataframe with the time shift
        """

        # First get X range [ the tested locations for the time shift ]
        tmp_x = self.model.time_reparameterization(self.x_torch)[0]
        # X = np.zeros((tmp_x.shape[0], self.N_biomarkers))
        x_min = np.inf
        x_max = -np.inf
        for biom in range(self.N_biomarkers):
            x_biom = ((self.model.time_reparameterization(self.x_torch)[biom]).detach().data.cpu().numpy().squeeze())
            if np.nanmin(x_biom) < x_min:
                x_min = np.nanmin(x_biom)
            if np.nanmax(x_biom) > x_max:
                x_max = np.nanmax(x_biom)

        x_delta = float((x_max - x_min) / 30)
        x_range = Variable(torch.arange(x_min, x_max, x_delta))
        x_range = x_range.unsqueeze(1)

        df_data = df_input.copy()

        # The X mean and std is the same for every biom.
        df_data[["NORM_Time"]] = (df_data[[f'{time_var}']] - self.x_mean_std[0][0]) / self.x_mean_std[0][1]
        x_shifted = x_range.data.cpu().numpy().squeeze() + df_data[list(('NORM_Time',) * len(x_range))]

        df_prob = pd.DataFrame(data=[], index=np.arange(x_range.shape[0] * df_data.shape[0]), columns=self.names_biomarkers)        
        df_prob["Tested_Pos"] = np.asarray((x_range.squeeze().data.numpy(),) * df_data.shape[0]).reshape(-1)
        df_prob["Instance"] = np.asarray((np.arange(df_data.shape[0]),) * len(x_range)).T.reshape(-1)
        df_prob[f"{id_var}"] = np.asarray((df_input[f"{id_var}"],) * len(x_range)).T.reshape(-1)
        df_prob = df_prob.set_index([f'{id_var}', 'Instance', 'Tested_Pos']).sort_index(level=[f'{id_var}', 'Instance', 'Tested_Pos']).copy()

        #TODO: Set to np.nan the prob for which x_shifted is outside the limits of the training
        for biom_pos, biom_id in enumerate(self.names_biomarkers):
            prob = None
            num_iters = 200
            for i in range(num_iters):  # Compute this 200 times
                sc_biom = (df_data[[f'{biom_id}']] - self.y_mean_std[biom_pos][0]) / self.y_mean_std[biom_pos][1]
                pred_test = x_shifted.apply(lambda x: self.model.branches[biom_pos][1](
                    Variable(torch.Tensor(x.values)).to(self.device).unsqueeze(1))[:, 0].data.numpy(), axis=0)
                noise = self.model.branches[self.N_biomarkers].sigma[biom_pos].detach().data.cpu().numpy()
                error = ((pred_test - sc_biom.values) ** 2 / np.exp(noise))
                if prob is None:
                    prob = .5 * (1 * noise + error) / num_iters
                else:
                    prob += .5 * (1 * noise + error) / num_iters
            tmp_prob = prob.copy()
            tmp_prob.columns = x_range.squeeze().data.numpy()
            tmp_prob = pd.melt(tmp_prob.T.reset_index(), id_vars='index').copy()
            tmp_prob.rename(columns={'index': 'Tested_Pos', 'variable': f'Instance'}, inplace=True)
            tmp_prob[[f'{id_var}']] = df_prob.reset_index()[[f'{id_var}']].values
            tmp_prob = tmp_prob.set_index([f'{id_var}', 'Instance', 'Tested_Pos']).sort_index(
                level=[f'{id_var}', 'Instance', 'Tested_Pos']).copy()

            df_prob[[biom_id]] = tmp_prob.values

        # Sum along the biomarkers
        df_prob_sum = df_prob.sum(axis=1, skipna=True)
        # Get minimum position for each instance
        min_time = df_prob_sum.reset_index().groupby(['RID', 'Tested_Pos']).sum().drop('Instance', axis=1).reset_index(level='RID').groupby(['RID'])[0].idxmin()
        # Since the values of the axis are already x_range, we can scale them directly
        optim_time = min_time * self.x_mean_std[0][1] + self.x_mean_std[0][0]
        optim_time = optim_time.reset_index().set_index('RID')
        optim_time.rename(columns={0: 'Time Shift'}, inplace=True)

        # df_data
        output_time = df_data[[f'{id_var}', f'{time_var}']].set_index(f"{id_var}")
        output_time[["Time Shift"]] = optim_time[["Time Shift"]].copy()
        output_time = output_time[~output_time.index.duplicated(keep='first')]
        # optim_time[[f'{id_var}']] = df_data[[f'{id_var}']].values
        # optim_time[[f'{time_var}']] = df_data[[f'{time_var}']].values
        # optim_time.drop('Instance', inplace=True, axis=1)

        # return optim_time
        return output_time
   

    def Diagnostic_predictions(self, predictions, verbose = True, group = [], save_fig = ''):
        x_min = float('inf')
        x_max = float('-inf')
        for bio_pos,biomarker in enumerate(range(self.N_biomarkers)):
            x_data = ((self.model.time_reparameterization(self.x_torch)[biomarker]).detach().data.cpu().numpy())
            if (float(np.min(x_data)) < x_min):
                x_min = float(np.min(x_data))
            if (float(np.max(x_data)) > x_max):
                x_max = float(np.max(x_data))

        x_range = Variable(torch.arange(x_min,x_max, float((x_max-x_min)/30)))
        x_range = x_range.reshape(x_range.size()[0],1)

        optimum = []

        for i in range(len(predictions)):
            print('--- Subject ' + str(i))
            min_time_pos = np.nanargmin(np.nansum(predictions[i],0))
            min_time = ((x_range.numpy()[min_time_pos])[0])
            if verbose:
                print('Max log-likelihood at time: ' + str(min_time * self.x_mean_std[0][1] + self.x_mean_std[0][0]))
            optimum.append(min_time * self.x_mean_std[0][1] + self.x_mean_std[0][0])
            if verbose:
                for biom in range(self.N_biomarkers):
                    min_time_pos_biom = np.nanargmin(np.hstack(predictions[i][biom]))
                    print('### Biomarker ' + str(biom) + ' max log-lik time: ' + str((x_range.numpy()[min_time_pos_biom])* self.x_mean_std[0][1] + self.x_mean_std[0][0]))

        if len(group)>0:
            p_df = pd.DataFrame({"class": group, "vals": optimum})
            fig, ax = plt.subplots(figsize=(8,6))
            for label, df in p_df.groupby('class'):
                if len(df.vals)>1:
                    df.vals.plot(kind="kde", ax=ax, label= label)
                else:
                    print('Warning: ' + label + ' group has only 1 element and will not be displayed')
            plt.legend()

            if save_fig:
                plt.savefig(save_fig + '/time_shift_distribution.png' )
            else:
                plt.show()   
 
        return(optimum)       

    def Threshold_to_time(self, threshold, list_biom = [], save_fig = '', from_EBM = False):
        if len(list_biom)<1:
            list_biom = self.names_biomarkers
        if np.isscalar(threshold):
            value = threshold
            threshold = []
            for i in range(len(list_biom)):
                threshold.append(value)

        x_min = float('inf')
        x_max = float('-inf')

        y_min = float('inf')
        y_max = float('-inf')

        for bio_pos,biomarker in enumerate(list_biom):
            bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
            x_data = (self.model.time_reparameterization(self.x_torch)[bio_id]).detach().data.cpu().numpy()
            y_data = (self.y_torch[bio_id]).detach().data.cpu().numpy()
            if (float(np.min(x_data)) < x_min):
                x_min = float(np.min(x_data))
            if (float(np.max(x_data)) > x_max):
                x_max = float(np.max(x_data))

            if (float(np.min(y_data)) < y_min):
                y_min = float(np.min(y_data))
            if (float(np.max(y_data)) > y_max):
                y_max = float(np.max(y_data))

        x_range = Variable(torch.arange(x_min,x_max, float((x_max-x_min)/50)))
        x_range = x_range.reshape(x_range.size()[0],1)

        new_x = self.Transform_subjects()
        time_dist = []
        for bio_pos,biomarker in enumerate(list_biom):
            bio_id = np.where([self.names_biomarkers[i]==biomarker for i in range(self.N_biomarkers)])[0][0]
            time_dist.append([])
            if not from_EBM:
                scaled_thresh = (threshold[bio_pos]-self.y_mean_std[bio_pos][0])/ self.y_mean_std[bio_pos][1]
            else:
                scaled_thresh = threshold[bio_pos]
            time_passed = []
            for i in range(200):
                predictions = self.model.branches[bio_id][1](x_range)[:,0].data.cpu().numpy()
                if len(np.where(predictions>scaled_thresh)[0])>0:
                    position = x_range[int(np.where(predictions>scaled_thresh)[0][0])].data.cpu().numpy()
                else:
                    position = x_range[-1]
                time_passed.append(position * self.x_mean_std[bio_pos][1] + self.x_mean_std[bio_pos][0])
            if save_fig:
                max_y = 0
                min_y = 0
                time_dist[bio_pos].append(np.array([time_passed[i][0] for i in range(200)]))
        mean_values = [np.mean(time_dist[i]) for i in range(len(list_biom))]
        order_biom = np.argsort(mean_values)
        if save_fig:
            plot_time_dist = [time_dist[i][0] for i in order_biom]
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.tight_layout()
            df = pd.DataFrame(plot_time_dist, index=[list_biom[i] for i in order_biom])
            df.T.boxplot(vert=False)
            plt.subplots_adjust(left=0.25)
            plt.xlabel('time')
            if not from_EBM:
                plt.title('time to reach threshold') 
                plt.savefig(save_fig + '/time_to_threshold' + '.png' )
            else:
                plt.title('time to reach threshold (EBM ordering)')
                plt.savefig(save_fig + '/time_to_threshold_EBM' + '.png' )
            plt.close()
            
            results = [[np.mean(time_dist[i]), np.std(time_dist[i]), list_biom[i]] for i in order_biom]
            if not from_EBM:
                pd.DataFrame(results,columns=['avg time','std time','biomarker name']).to_csv(save_fig +'/time_dist_thr' + '.csv')
            else:
                pd.DataFrame(results,columns=['avg time (EBM ordering)','std time','biomarker name']).to_csv(save_fig +'/time_dist_thr_EBM' + '.csv')
        if not from_EBM:
            return results

    def EBM_style_plot(self, group, list_biom = []):
        from sklearn import discriminant_analysis
        lda_classifier = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1, shrinkage = None, store_covariance = True)

        if len(list_biom)<1:
            list_biom = self.names_biomarkers 
       
        threshold = []
        group_values = np.unique(group)
        index_group0 = np.where(group==group_values[0])[0]
        index_group1 = np.where(group==group_values[1])[0]

        for bio_pos,biomarker in enumerate(list_biom):
            group0 = [self.y[bio_pos][i][0] for i in index_group0]
            group1 = [self.y[bio_pos][i][0] for i in index_group1]
            data = np.hstack([group0,group1]).reshape(-1,1)
            labels = np.hstack([np.repeat(0,len(group0)),np.repeat(1,len(group1))])                        
            lda_classifier.fit(data,labels)
            separating_plane = -lda_classifier.intercept_/lda_classifier.coef_ 
            #print(biomarker +' '+ str(separating_plane[0] *  self.y_mean_std[bio_pos][1] + self.y_mean_std[bio_pos][0] ))
            threshold.append(separating_plane[0])
            plt.boxplot([group0, group1])
            plt.hlines(threshold[bio_pos],-10,10)
            plt.savefig('./separating_plane_' + biomarker + '.png' )
            plt.close()
        self.Threshold_to_time(threshold = threshold, list_biom = list_biom, save_fig = './', from_EBM = True)


    def Return_time_parameters(self):
        return self.tr.time_parameters.detach().data.cpu().numpy()

    def Save_predictions(self,prediction,path):
        with open(path, "wb") as f:
            pickle.dump(prediction, f)

    def Load_predictions(self,path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(path, map_location = device)
        return data

    def Transform_subjects(self):
        data = self.model.time_reparameterization(self.x_torch)
        output_x = []
        for biom in range(self.N_biomarkers):
            output_x.append([])
            data[biom] = data[biom].detach()
            len_measurements = [len(self.x_torch[biom][i]) for i in range(len(self.x_torch[0]))]
            for i in range(len(len_measurements)):
                output_x[biom].append(data[biom][sum(len_measurements[:i]):sum(len_measurements[:i]) + len_measurements[i]].data.cpu().numpy())
        return output_x

    def Save(self, path):
        torch.save(self.model, path + '/model_.pth')
        torch.save(self.tr, path + '/tr_.pth')
        data = [self.x, self.y, self.N_biomarkers, self.N_subs, self.names_biomarkers, self.groups, self.group_name, self.reparameterization_model, self.trade_off, self.x_mean_std, self.y_mean_std]
        torch.save(data, path + '/misc.dat')

    def Load(self, path):
        try:
            torch._utils._rebuild_tensor_v2
        except AttributeError:
            def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
                tensor.requires_grad = requires_grad
                tensor._backward_hooks = backward_hooks
                return tensor

            torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load .dat
        data = torch.load(path + '/misc.dat', map_location=device)
        self.x = data[0]
        self.y = data[1]
        self.N_biomarkers = data[2]
        self.N_subs = data[3]
        self.names_biomarkers = data[4]
        self.groups = data[5]
        self.group_name = data[6]
        self.reparameterization_model = data[7]
        self.trade_off = data[8]
        self.x_mean_std = data[9]
        self.y_mean_std = data[10]

        # Load .pth
        self.model = torch.load(path + '/model_.pth', map_location=device)
        self.tr = torch.load(path + '/tr_.pth', map_location=device)

        # Match devices - e.g. if model was trained on GPU but one wants to load on CPU, this allows to match devices
        self.device = device
        self.model.time_reparameterization.device = device
        self.tr.device = device
        for bio_pos, biomarker in enumerate(self.names_biomarkers):
            bio_id = np.where([self.names_biomarkers[i] == biomarker for i in range(self.N_biomarkers)])[0][0]
            self.model.branches[bio_id][1].device = device

        self.Initialize(first=False)


def Cost(target, predicted, predicted2, monotonicity, trade_off, device):

    output_x = predicted[0]
    total_ll = 0
    constraint = 0
    N_biomarkers = len(output_x)

    for i in range(N_biomarkers):
        x = output_x[i][:,0]
        Dx = predicted2[i][:,1]
#        Dx = predicted2[0][i][:,1]
        sigma = predicted[1][i]

        total_ll += -0.5 * (x.size(0)*(torch.log(sigma)) + torch.sum((target[i] - x) ** 2)/(sigma))

        monotonicity_Dx = float(monotonicity[i]) * Dx

        # x_relu = - trade_off * F.relu(-monotonicity_Dx.to(device))

        x_relu = - torch.logsumexp(torch.cat([torch.zeros(monotonicity_Dx.size(0)).to(device), - trade_off * monotonicity_Dx.to(device)],0),0)

        constraint += torch.sum(x_relu)


    return total_ll, constraint

class Regression_Model(nn.Module):
    def __init__(self, Time_reparameterization, N_subjects, N_biomarkers, device, N_rf = 20 , seed = int(1), l = 2, sigma = 1 , init_noise = 1):
        super(Regression_Model, self).__init__()
        self.N_subjects = N_subjects
        self.N_biomarkers = N_biomarkers
        self.time_reparameterization = Time_reparameterization
        GP_list = [nn.Sequential(OrderedDict([('time',self.time_reparameterization),('GP'+str(i),GP(1, 1, N_rf, seed, l, sigma, device))])) for i in range(N_biomarkers)]
        GP_list.append(Gauss_NoiseModel(self.N_biomarkers, init_noise))
        GP_list.append(Random_effects(self.N_subjects,self.N_biomarkers, device))
        self.branches = nn.ModuleList(GP_list)

        for i, branch in enumerate(self.branches):
            self.add_module(str(i), branch)

    def forward(self, x):
        output = []
        for i in range(self.N_biomarkers):
            output_i = self.branches[i][0](x)[i]
            output_ranef_i = self.branches[self.N_biomarkers+1](x)[i]  #computing random effects
            output.append(self.branches[i][1](output_i) + output_ranef_i)  #adding fixed and random effects
        return output, self.branches[self.N_biomarkers](x)

    def KL(self):
        KL_tot = 0
        for i in range(len(self.branches[0])):
            KL_tot += torch.sum(self.branches[i][1].KL())
        return KL_tot

class Random_effects(nn.Module):
    def __init__(self, N_subjects, N_biomarkers, device):
        super(Random_effects, self).__init__()
        self.N_subjects = N_subjects
        self.N_biomarkers = N_biomarkers
        self.ranef_inter = nn.Parameter(torch.Tensor(N_biomarkers, self.N_subjects).fill_(-2), requires_grad=True)
        self.ranef_slope = nn.Parameter(torch.Tensor(N_biomarkers, self.N_subjects).fill_(-2), requires_grad=True)
        self.device = device

    def set_index(self, list_id):
        return list_id

    def forward(self,x):
        output_x = []
        for biom in range(self.N_biomarkers):
            len_measurements = [len(x[biom][i]) for i in range(len(x[biom]))]
            sampler_inter = Variable(torch.randn(1, len(x[0])), requires_grad=False).type(torch.FloatTensor).to(self.device)
            sampler_slope = Variable(torch.randn(1, len(x[0])), requires_grad=False).type(torch.FloatTensor).to(self.device)
            matrix_inter = Variable(torch.zeros(len(len_measurements), sum(len_measurements))).to(self.device)
            matrix_slope = Variable(torch.zeros(len(len_measurements), sum(len_measurements))).to(self.device)

            for i in range(len(len_measurements)):
                if len_measurements[i]>2:
                    matrix_inter[i,sum(len_measurements[:i]):sum(len_measurements[:i])+len_measurements[i]] = \
                          matrix_inter[i,sum(len_measurements[:i]):sum(len_measurements[:i])+len_measurements[i]] + 1
                if len_measurements[i]>5:
                    matrix_slope[i,sum(len_measurements[:i]):sum(len_measurements[:i])+len_measurements[i]] = \
                          matrix_slope[i,sum(len_measurements[:i]):sum(len_measurements[:i])+len_measurements[i]] + 1

            w_inter = sampler_inter * torch.exp(self.ranef_inter[biom,:])[self.set_index(self.list_id)]
            w_inter_expanded = w_inter.resize(1,len(x[0])).mm(matrix_inter)
            w_slope = sampler_slope * torch.exp(self.ranef_slope[biom,:])[self.set_index(self.list_id)]
            w_slope_expanded = w_slope.resize(1,len(x[0])).mm(matrix_slope)
            output_x.append((Variable(torch.cat(x[biom])).to(self.device) * w_slope_expanded + w_inter_expanded).resize(w_slope_expanded.size(1),1))
        return output_x

class Time_reparameterization(nn.Module):
    def __init__(self, N_subjects, N_biomarkers, reparametrization_type, device):
        super(Time_reparameterization, self).__init__()
        self.N_biomarkers = N_biomarkers
        self.reparametrization_type = reparametrization_type
        if self.reparametrization_type == 'time_shift':
            self.time_parameters = nn.Parameter(torch.Tensor(N_subjects).fill_(0), requires_grad=True)
        if self.reparametrization_type == 'linear':
            self.time_parameters0 = nn.Parameter(torch.Tensor(N_subjects, 1).fill_(0), requires_grad=True)
            self.time_parameters1 = nn.Parameter(torch.Tensor(N_subjects, 1).fill_(1), requires_grad=True)

        self.device = device

    def set_index(self, list_id):
        return list_id

    def forward(self, x):
        output_x = []
        for biom in range(self.N_biomarkers):
            len_measurements = [len(x[biom][i]) for i in range(len(x[biom]))]
            matrix = Variable(torch.zeros(len(len_measurements), sum(len_measurements))).to(self.device)
            for i in range(len(len_measurements)):
                matrix[i, sum(len_measurements[:i]):sum(len_measurements[:i]) + len_measurements[i]] = \
                    matrix[i, sum(len_measurements[:i]):sum(len_measurements[:i]) + len_measurements[i]] + 1

            if self.reparametrization_type == 'time_shift':
                time_expanded = self.time_parameters[self.set_index(self.list_id)].resize(1, self.time_parameters[self.set_index(self.list_id)].size()[0]).mm(matrix)
                output_x.append((Variable(torch.cat(x[biom])).to(self.device) + time_expanded).resize(time_expanded.size(1), 1))
            if self.reparametrization_type == 'linear':
                time_expanded_0 = self.time_parameters0[self.set_index(self.list_id)].resize(1, self.time_parameters0[self.set_index(self.list_id)].size()[0]).mm(matrix)
                time_expanded_1 = self.time_parameters1[self.set_index(self.list_id)].resize(1, self.time_parameters1[self.set_index(self.list_id)].size()[0]).mm(matrix)

                output_x.append((Variable(torch.cat(x[biom])).to(self.device) * time_expanded_1 + time_expanded_0).resize(time_expanded_1.size(1), 1))

        return output_x

class Gauss_NoiseModel(nn.Module):
    def __init__(self, out_size, init):
        super(Gauss_NoiseModel, self).__init__()  
        self.sigma = nn.Parameter(torch.Tensor(out_size).fill_(init), requires_grad=True)
    def forward(self, x):
        return torch.exp(self.sigma)

class GP(nn.Module):
    def __init__(self, input_dim, output_dim, N_rf, seed, input_l, input_sigma, device, prior_fixed=True, var_fixed=False, order = 1, level = 0):
        super(GP, self).__init__()  

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N_rf = N_rf
        self.prior_fixed = prior_fixed
        self.input_l = input_l
        self.input_sigma = input_sigma

        self.seed = seed
        self.order = order
        self.level = level

        self.device = device

        self.l = nn.Parameter(torch.log(torch.Tensor(input_dim).fill_(self.input_l)), requires_grad=True)
        self.sigma = nn.Parameter(torch.Tensor([[self.input_sigma]]), requires_grad=True)

        self.m_omega = nn.Parameter(torch.Tensor(self.input_dim, self.N_rf), requires_grad=True)
        self.s_omega = nn.Parameter(torch.Tensor(self.input_dim, self.N_rf), requires_grad=True)
        self.m_omega.data = torch.rand(self.input_dim, self.N_rf, device = device) - torch.FloatTensor([[0.5]]).to(device)
        self.s_omega.data = torch.rand(self.input_dim, self.N_rf, device = device) - torch.FloatTensor([[0.5]]).to(device)

        self.m_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, self.output_dim), requires_grad=True)
        self.s_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, self.output_dim), requires_grad=True)
        self.m_w.data = 2 * torch.rand(2*self.N_rf + 1, self.output_dim, device = device) - torch.FloatTensor([[1.]]).to(device)
        self.s_w.data = torch.rand(2*self.N_rf + 1, self.output_dim, device = device) - torch.FloatTensor([[0.5]]).to(device)

    def forward(self, x):
            if self.order == 1:
                if self.level > 0:
                    input_x = x[:, :self.input_dim].to(self.device)
                    Dinput_x = x[:, self.input_dim:].to(self.device)
                else:
                    # Assuming time as model input
                    input_x = x.to(self.device)
                    Dinput_x = Variable(torch.ones(x.size(0), x.size(1)), requires_grad=False).type(torch.FloatTensor).to(self.device)

            if self.prior_fixed:
                exp_l = 1 / torch.exp(self.l)
                exp_sigma = torch.exp(self.sigma)
                rng = np.random.RandomState(self.seed)
                sampler = Variable(torch.from_numpy(rng.randn(self.input_dim, self.N_rf)), requires_grad=False).type(
                    torch.FloatTensor).to(self.device)
                omega = Variable(torch.zeros(self.input_dim, self.N_rf)).to(self.device)
                for p in range(len(self.l)):
                    omega[p] = torch.sqrt(exp_l[p]) * sampler[p]
                #omega = torch.sqrt(torch.exp(self.s_omega)) * sampler + self.m_omega

                sqrt_len_omega = Variable(torch.sqrt(torch.FloatTensor([self.N_rf]))).to(self.device)


                if self.order == 0:
                    Phi = (torch.sqrt(exp_sigma) / sqrt_len_omega) * torch.cat(
                            [torch.cos(torch.mm(x, omega)), torch.sin(torch.mm(x, omega))], dim=1)
                elif self.order == 1:
                    Phi = (torch.sqrt(exp_sigma) / sqrt_len_omega) * \
                               torch.cat([torch.cos(torch.mm(input_x, omega)), torch.sin(torch.mm(input_x, omega)),
                                          -torch.mul(torch.sin(torch.mm(input_x, omega)), torch.mm(Dinput_x, omega)),
                                          torch.mul(torch.cos(torch.mm(input_x, omega)), torch.mm(Dinput_x, omega))],
                                         dim=1)

            exp_s_w = torch.exp(self.s_w)
            sampler = Variable(torch.randn(2*self.N_rf + 1, self.output_dim), requires_grad=False).type(torch.FloatTensor).to(self.device)
            W = torch.sqrt(exp_s_w) * sampler + self.m_w

            if self.order == 0:
                return torch.mm(Phi, W[1:,:]) + W[0,:]
            elif self.order == 1:
                product = torch.mm(Phi[:, : 2*self.N_rf], W[1:,:])
                Dproduct = torch.mm(Phi[:, 2*self.N_rf:], W[1:,:])
                return torch.cat([product, Dproduct], dim=1)


    def KL(self):
        KL = 0
        KL += 0.5*torch.sum(torch.exp(self.s_w) + self.m_w**2 - torch.log(torch.exp(self.s_w)) - 1)

        return(KL)

    def Reset(self):
        self.l = nn.Parameter(torch.log(torch.Tensor(self.input_dim).fill_(self.input_l)), requires_grad=True).to(self.device)
        self.sigma = nn.Parameter(torch.Tensor([[self.input_sigma]]), requires_grad=True).to(self.device)

#        self.m_omega = nn.Parameter(torch.Tensor(self.input_dim, self.N_rf), requires_grad=True)
#        self.s_omega = nn.Parameter(torch.Tensor(self.input_dim, self.N_rf), requires_grad=True)
#        self.m_omega.data = torch.rand(self.input_dim, self.N_rf, device = self.device) - torch.FloatTensor([[0.5]]).to(self.device)
#        self.s_omega.data = torch.rand(self.input_dim, self.N_rf, device = self.device) - torch.FloatTensor([[0.5]]).to(self.device)

#        self.m_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, self.output_dim), requires_grad=True)
#        self.s_w = nn.Parameter(torch.Tensor(2*self.N_rf + 1, self.output_dim), requires_grad=True)
#        self.m_w.data = 2 * torch.rand(2*self.N_rf + 1, self.output_dim, device = self.device) - torch.FloatTensor([[1.]]).to(self.device)
#        self.s_w.data = torch.rand(2*self.N_rf + 1, self.output_dim, device = self.device) - torch.FloatTensor([[0.5]]).to(self.device)

class LinModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinModel, self).__init__()  # always call parent's init
        self.linear = nn.Linear(in_size, out_size)  # layer parameters

    def forward(self, x):
        return self.linear(x)

def convert_from_df(table, list_biomarkers, time_var = 'Time'):
    X = []
    Y = []

    # list of individuals
    list_RID = np.unique(table[['RID']])

   # list of biomarkers
#    list_biomarkers = table.columns[range(2, len(table.columns))]

    RID = []

    for id_biom, biomarker in enumerate(list_biomarkers):
        X.append([])
        Y.append([])

    # Parsing every biomarker and assigning to the list
    for id_sub, sub in enumerate(list_RID):
        flag_missing = 0
        for id_biom, biomarker in enumerate(list_biomarkers):
            indices = np.where(np.in1d(table.RID, sub))[0]
            X[id_biom].append(np.array(table[[time_var]])[np.where(np.in1d(table.RID, sub))[0]].flatten())
            Y[id_biom].append(np.array(table[[biomarker]])[np.where(np.in1d(table.RID, sub))[0]].flatten())

            idx_to_remove = ~np.isnan(Y[id_biom][id_sub])

            Y[id_biom][id_sub] = Y[id_biom][id_sub][idx_to_remove]
            X[id_biom][id_sub] = X[id_biom][id_sub][idx_to_remove]

            if len(Y[id_biom][id_sub]) < 1:
                flag_missing = flag_missing + 1

        if flag_missing == 0:
            RID.append(sub)

    Xtrain = []
    Ytrain = []

    for id_biom, biomarker in enumerate(list_biomarkers):
        Xtrain.append([])
        Ytrain.append([])
    for id_sub, sub in enumerate(list_RID):
        if np.in1d(sub, RID)[0]:
            for id_biom, biomarker in enumerate(list_biomarkers):
                Xtrain[id_biom].append(X[id_biom][id_sub])
                Ytrain[id_biom].append(Y[id_biom][id_sub])

    group = []

    # check if group information is present
    if len(np.where('group' == table.columns)[0])>0:
        for sub in RID:
            group.append(table.group[np.where(np.in1d(table.RID,sub))[0][0]])

    group = np.array(group)

    return Xtrain, Ytrain, RID, list_biomarkers, group

def partition(n_in, n):
    import random
    list_in = random.sample(list(range(n_in)), n_in)
    division = len(list_in) / float(n)

    return [ list_in[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]
