#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary source code attached for the manuscript:

"Determination and validation of a design space for mesenchymal stem cell cultivation processes using prediction intervals" by
Keita Hirono, Yusuke Hayashi, Isuru A. Udugama, Mohamed Rami Gaddem, Kenjiro Tanaka, Yuto Takemoto, Ryuji Kato, Masahiro Kino-oka, Hirokazu Sugiyama

Last saved on Jan 23 2025


Lines     : Functions                                         (Corresponding section)
370 - 435 : Impacts of sample size on model prediction        (Supplementary Result)
440 - 490 : Impacts of sample size on design space validation (Supplementary Result)

Created by Keita Hirono/Sugiyama-Badr Lab/The University of Tokyo
"""

import itertools
import numpy as np
import pandas as pd; pd.options.mode.copy_on_write = True
from matplotlib import pyplot as plt
import statistics
from scipy import stats
import scipy.optimize as optimize
from scipy.integrate import solve_ivp
from main import calc_ode, calc_nrmse, calc_mse, plot_nrmse
from multiprocessing import Pool
import scipy.ndimage

def simu_growth_R1(var, k, para, dv):
    sl = 0.1
    mc_period, day_passage, r_mc, mc_in = dv['mc_period'], dv['day_passage'], dv['r_mc'], dv['mc_in']
    f = lambda t, var: calc_ode(t, var, k, para)
    
    Xa_cal_list = []; Glc_cal_list = []; Lac_cal_list = []
    t_cal_list = []
    Glc_in, Lac_in = mc_in
    
    var_init = var
    if 1 <= mc_period <= day_passage:
        t_list = np.arange(24, mc_period * 24 + sl, sl)
    var_list = solve_ivp(f, (t_list[0], t_list[-1]), var_init, method='RK45', t_eval = t_list)
    t_list = var_list.t
    var_list = var_list.y
    
    # record
    Xa_cal_list.extend(var_list[0,:])
    Glc_cal_list.extend(var_list[1,:])
    Lac_cal_list.extend(var_list[2,:])
    t_cal_list.extend(t_list)
    
    if mc_period == 1:
        Xa_cal_list.pop(-1)
        Glc_cal_list.pop(-1)
        Lac_cal_list.pop(-1)
        t_cal_list.pop(-1)
        
    i = 1
    while i * mc_period < day_passage:         
        if i == 1 and mc_period == 1:
            pass
        else:
            var_init = [var_list[0][-1], 
                    var_list[1][-1] * (1 - r_mc[i-1]) + r_mc[i-1] * Glc_in, 
                    var_list[2][-1] * (1 - r_mc[i-1]) + r_mc[i-1] * Lac_in, 
                    ]
        if (i + 1) * mc_period >= day_passage:
            t_list = np.arange(mc_period * i * 24 + sl, day_passage * 24 + 0.1 * sl, sl)
        else:  
            t_list = np.arange(mc_period * i * 24 + sl, mc_period * (i + 1) * 24 + 0.1 * sl, sl)
        var_list = solve_ivp(f, (t_list[0], t_list[-1]), var_init, method='RK45', t_eval = t_list)
        t_list = var_list.t
        var_list = var_list.y
        
        # record
        Xa_cal_list.extend(var_list[0,:])
        Glc_cal_list.extend(var_list[1,:])
        Lac_cal_list.extend(var_list[2,:])
        t_cal_list.extend(t_list)
        
        i += 1
    
    cal_list = [Xa_cal_list, Glc_cal_list, Lac_cal_list, t_cal_list]
    return cal_list


def calc_error_R1(k, df_F, df_IN, para, figs, dv, n_sample=1, graph=True, errorbar=True):
    mse_list = []
    nrmse_ave = [0 for _ in range(n_sample)]
    ss = 0
    
    for i in range(n_sample):
        sample = df_F.columns[i+1]
        
        x_max = df_IN['xmax'][i]
        para[0] = x_max
        
        epsilon = df_IN['epsilon'][i]
        para[1] = epsilon
        
        alpha = df_IN['alpha'][i]
        X0_i = df_F[sample][0] * alpha
        para[2] = 0
        
        var = [X0_i, 5, 2]
        y = simu_growth_R1(var, k, para, dv)
        
        pred_list_1 = y[0][::60]
        t_true = df_F['time'][1:]
        X_true = df_F[sample][1:] / dv['S']
        
        mse_batch = []
        actual = np.array(X_true, dtype=float)
        predicted = np.array(pred_list_1, dtype=float)
        mse = calc_mse(actual, predicted)
        mse_batch.append(mse)
        mse_list.append(sum(mse_batch))
        
        if graph:
            nrmse_ave = plot_nrmse(df_F, mse_batch, nrmse_ave, n_sample, sample, t_true, X_true, y, dv['S'])
            
        ss += np.sum(np.square(actual - predicted))
    error = calc_nrmse(df_F, mse_list, dv['S'])
    
    if errorbar:
        return error
    else:
        return ss


def f(x):
    ##FIXED DESIGN FOR EXP1
    S = 1.53 * 1.53       
    V = 0.2 * S / 1000  
    MC_in = [5, 2]

    ##FIXED PARAMS
    Q_glc = 1.078e-10       # mmol/(cell h)
    Q_lac = Q_glc * 2       # mmol/(cell h)
    K_sd = 24.7

    PARA = [0, 0, 0, K_sd, Q_glc, Q_lac, S, V]

    ##FIXED OPERATION FOR EXP2
    Day_passage = 9
    MC_period = 9 
    R_mc = [0.5 for _ in range((Day_passage-1) // MC_period)]
    dv = dict(mc_period=MC_period, day_passage=Day_passage, r_mc=R_mc, mc_in=MC_in, S=S)

    k_opt = [4.023e-02]
    
    ##MAIN FUNCTION
    fit, names = x[1:]
    res_fit = [[], [], []]
    res_val = [[], [], []]
    
    ##DATASET
    df = pd.read_csv('csv/Exp2_measured_data.csv')
    df_IN = pd.read_csv('csv/Exp2_calculated_parameters.csv', index_col=0)
    df0_IN = pd.read_csv('csv/Exp1_calculated_parameters.csv', index_col=0)
    
    df_fit = df.filter(items=(['time'] + names), axis=1)
    df_in = df_IN.filter(items=names, axis=0)
    
    n_sample = df_fit.shape[1] - 1
    k2_opt_list = []
    
    #PARAMETER ESTIMATION FOR EVERY SAMPLE
    for j in range(n_sample):
        figs = dict(name='stable1/stable1_idx_{}_fit_{}'.format(fit, j+1), style='--', color='r')
        
        n0 = n_sample
        n_sample = 1
        
        df_fit_PI = df_fit.iloc[:,[0,j+1]]
        df_in_PI = pd.DataFrame([df_in.iloc[j,:]])
        
        graph = False; errorbar = False
        mi_nrmse = optimize.minimize(calc_error_R1, k_opt, args=(df_fit_PI, df_in_PI, PARA, figs, dv, n_sample, graph, errorbar),
                                        method='Nelder-Mead', 
                                        options={'maxiter':5000})
        k2_opt = mi_nrmse.x
        k2_opt_list.append(k2_opt[0])
        
        n_sample = n0

    mean = statistics.mean(k2_opt_list)
    
    for j in range(3):
        figs = dict(name='fig3a_idx_{}_{}_fit'.format(fit, str(15*(j+1))), style='--', color='r')
        df_in_i = df_in.filter(like=str(1500*(j+1)), axis=0) 
        df_fit_i = pd.concat([df['time'], df_fit.filter(like=str(1500*(j+1)), axis=1)], axis=1) 
        n_sample = df_fit_i.shape[1] - 1
        if n_sample == 0: continue
        graph = False
        nrmse_F2 = calc_error_R1([mean], df_fit_i, df_in_i, PARA, figs, dv, n_sample, graph)              
        res_fit[j].append(nrmse_F2)
        
    df_in = df_IN.loc[~df_IN.index.str.contains(fit)]
    df_all = pd.concat([df0_IN, df_IN.filter(items=names, axis=0)], axis=0)
    df_in.iloc[:,:] = df_all.mean()
    
    df_val = df.loc[:, ~df.columns.str.contains(fit)]
    for j in range(3):
        figs = dict(name='fig3b_idx_{}_{}_val'.format(fit, str(15*(j+1))), style='-', color='tab:orange')
        df_val_i = pd.concat([df['time'], df_val.filter(like=str(1500*(j+1)), axis=1)], axis=1)
        n_sample = df_val_i.shape[1] - 1
        graph = False
        nrmse_V2 = calc_error_R1([mean], df_val_i, df_in, PARA, figs, dv, n_sample, graph)     
        res_val[j].append(nrmse_V2)
    
    return res_fit, res_val
    

def f1(x):
    ##SETTING
    seed = 1234
    rng = np.random.default_rng(seed)
    n_ds = 1000

    ##QUALITY SPECIFICATION    
    CQA1 = 5e+4
    CQA2 = 0.8
    n_cpp1 = 9
    n_cpp2 = 33
    cpp1 = np.linspace(1500,4500,n_cpp1)
    cpp2 = np.linspace(9,1,n_cpp2)

    ##FIXED DESIGN FOR EXP1
    S = 1.53 * 1.53       
    V = 0.2 * S / 1000  
    MC_in = [5, 2]

    ##FIXED PARAMS
    Q_glc = 1.078e-10       # mmol/(cell h)
    Q_lac = Q_glc * 2       # mmol/(cell h)
    K_sd = 24.7

    PARA = [0, 0, 0, K_sd, Q_glc, Q_lac, S, V]

    ##FIXED OPERATION FOR EXP2
    Day_passage = 9
    MC_period = 9 
    R_mc = [0.5 for _ in range((Day_passage-1) // MC_period)]
    dv = dict(mc_period=MC_period, day_passage=Day_passage, r_mc=R_mc, mc_in=MC_in, S=S)

    k_opt = [4.023e-02]
    
    ##MAIN FUNCTION
    fit, names = x[1:]
    
    ##DATASET
    df = pd.read_csv('csv/Exp2_measured_data.csv')
    df_IN = pd.read_csv('csv/Exp2_calculated_parameters.csv', index_col=0)
    df0_IN = pd.read_csv('csv/Exp1_calculated_parameters.csv', index_col=0)
    x_test = [1500, 3000, 4500]
    
    df_fit = df.filter(items=(['time'] + names), axis=1)
    df_in = df_IN.filter(items=names, axis=0)
    
    n_sample = df_fit.shape[1] - 1
    k2_opt_list = []
    
    #PARAMETER ESTIMATION FOR EVERY SAMPLE
    for j in range(n_sample):
        figs = dict(name='stable1/stable1_idx_{}_fit_{}'.format(fit, j+1), style='--', color='r')
        
        n0 = n_sample
        n_sample = 1
        
        df_fit_PI = df_fit.iloc[:,[0,j+1]]
        df_in_PI = pd.DataFrame([df_in.iloc[j,:]])
        
        graph = False; errorbar = False
        mi_nrmse = optimize.minimize(calc_error_R1, k_opt, args=(df_fit_PI, df_in_PI, PARA, figs, dv, n_sample, graph, errorbar),
                                        method='Nelder-Mead', 
                                        options={'maxiter':5000})
        k2_opt = mi_nrmse.x
        k2_opt_list.append(k2_opt[0])
        
        n_sample = n0

    mean = statistics.mean(k2_opt_list)
    std = statistics.stdev(k2_opt_list)
    ta = stats.t.ppf(1-0.025, n_sample-1) 
    pi_lb, pi_ub = mean - ta * std * np.sqrt(1+1/n_sample), mean + ta * std * np.sqrt(1+1/n_sample) 
    
    df_in = df_IN.loc[~df_IN.index.str.contains(fit)]
    df_all = pd.concat([df0_IN, df_IN.filter(items=names, axis=0)], axis=0)
    df_in.iloc[:,:] = df_all.mean()
    
    dist_xmax = rng.choice(df_all['xmax'], n_ds)
    dist_eps = rng.choice(df_all['epsilon'], n_ds)
    dist_alp = rng.choice(df_all['alpha'], n_ds)

    res_ds = np.zeros((len(cpp2), len(cpp1)))
    
    for s in range(len(cpp1)):
        x_seed = cpp1[s]
    
        for x_max, epsilon, alpha in zip(dist_xmax, dist_eps, dist_alp):
            X0_i = x_seed * alpha
            para = [x_max, epsilon, 0, K_sd, Q_glc, Q_lac, S, V]
            var = [X0_i, 5, 2]
            
            y_lb = simu_growth_R1(var, [pi_lb], para, dv) 
            y_ub = simu_growth_R1(var, [pi_ub], para, dv) 
    
            for t in range(len(cpp2)):
                if (x_seed in x_test) or (cpp2[t] >= 4):
                    hr = cpp2[t] * 24
                    ind = [round(t, 1) for t in y_lb[-1]].index(hr)
                    
                    y1_lb, y1_ub = y_lb[0][ind]*S, y_ub[0][ind]*S
                    y2_lb, y2_ub = y_lb[0][ind]/x_max, y_ub[0][ind]/x_max
                    if CQA1 <= min(y1_lb, y1_ub) and max(y2_lb, y2_ub) < CQA2:
                        res_ds[t,s] += 100 / n_ds
        
    return res_ds
      

def f2(x):
    n_cpp2 = 33
    fit, pi, res_ds = x[1:]
    rec_obs = np.loadtxt('csv/Exp2{}_measured_probability_for_DS_validation.csv'.format(fit), delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[1,2,3], encoding="utf-8")
        
    tf_list = []
    for j in range(3):
        tfs = []
        for obs, cal in zip(rec_obs[:,j], res_ds[:,4*j]):
            if pi <= obs and cal < pi:  tfs.append('FN')
            if pi <= obs and pi <= cal: tfs.append('TP')
            if obs < pi and cal < pi:   tfs.append('TN')
            if obs < pi and pi <= cal:  tfs.append('FP')
        tf_list.append(tfs)
    
    tf_all = sum(tf_list, [])
    n = lambda x: tf_all.count(x)
    res_tfs = [n('TP'), n('FP'), n('TN'), n('FN')]
    
    return tf_list, res_tfs 


if __name__ == '__main__':
    
    ##SETTING
    cp = 16
    
    ##FIGURE SPECIFICATION
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    c_list = dict(TP='g', FP='r', TN='b', FN='orange')
    
    ##CALCULATION SPECIFICATION    
    n_cpp1 = 9
    n_cpp2 = 33
    cpp1 = np.linspace(1500,4500,n_cpp1)
    cpp2 = np.linspace(9,1,n_cpp2)
    
    x_test = [1500, 3000, 4500]
    t_test = np.linspace(9,1,33)
    
    es = 6
    sample = range(1,es+1)
    
    
    
    
    ##MODEL VALIDATION
    ###plot
    fig5, axes5 = plt.subplots(3, 3, figsize=(11,11), tight_layout=True)
    fig6, axes6 = plt.subplots(3, 3, figsize=(11,11), tight_layout=True)
    for axes in (axes5, axes6):
        for i in range(3):
            for j in range(3):
                axes[i,j].set_xlim(2,19)
                axes[i,j].set_ylim(0,15)
                axes[i,j].set_xticks(range(3,19,3))
                axes[i,j].axhline(10.0, c='g', linestyle='--')
                if i < 2:
                    axes[i,j].tick_params(labelbottom=False)
                if j > 0:
                    axes[i,j].tick_params(labelleft=False)

    
    for g, fit in enumerate(('A', 'B', 'C')):
        print(fit)
        
        ###data
        name_set = [['seed_{}00_{}_0{}'.format(15*(i+1), fit, j+1) for j in range(es)] for i in range(3)] 
        
        ##DATASET
        rec_obs = np.loadtxt('csv/Exp2{}_measured_probability_for_DS_validation.csv'.format(fit), delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[1,2,3], encoding="utf-8")
        t_test = np.loadtxt('csv/Exp2{}_measured_probability_for_DS_validation.csv'.format(fit), delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[0], encoding="utf-8")
        
        ###calculation
        for i in sample:
            b1 = list(itertools.combinations(name_set[0], i))
            b2 = list(itertools.combinations(name_set[1], i))
            b3 = list(itertools.combinations(name_set[2], i))
            print('{}C{} n={}'.format(len(name_set[0]),i,3*i))
            
            run = 0
            
            x_set = []
            for name1 in b1:
                for name2 in b2:
                    for name3 in b3:
                        # comb = '3'
                        names = list(name1) + list(name2) + list(name3)
                        x_set.append([i, fit, names])
                        run += 1
                        
            p = Pool(cp)
            res_pool = p.map(f, x_set)
            # --> res_fit, res_val
            
            res_fit = [[], [], []]
            res_val = [[], [], []]
            for k in range(len(res_pool)):
                for res, yk in zip((res_fit, res_val), res_pool[k]):
                    for l in range(3):
                        res[l].append(yk[l][0])
                    
            #NRMSE  
            for axes, res in zip((axes5, axes6), (res_fit, res_val)):
                for u in range(3):
                    if len(res[u]) > 2:
                        axes[g,u].errorbar(3*i, statistics.mean(res[u]), statistics.stdev(res[u]), capsize=6, fmt='o', markersize=8, ecolor='k', markeredgecolor='k', color='w')
                    else:
                        for v in range(len(res[u])):
                            axes[g,u].scatter(3*i, res[u][v], s=70, ec='k', color='None')
            print('--> {}runs'.format(run))
    plt.show()
    
    
    
    
    ##DS VALIDATION
    sample = range(4,6)
    x1_set = []
    for h in sample:
        for fit in ('A', 'B', 'C'):
            name_set = [['seed_{}00_{}_0{}'.format(15*(i+1), fit, j+1) for j in range(h)] for i in range(3)] 
            names = sum(name_set, [])
            x1_set.append([h, fit, names])
        
    p1 = Pool(cp)
    res1_pool = p1.map(f1, x1_set)
    # --> res_ds
    
    x2_set = []
    for ih, h in enumerate(sample):
        for i, fit in enumerate(('A', 'B', 'C')):
            res_ds = res1_pool[3*ih+i]
            for pi in (50, 70, 90):
                x2_set.append([h, fit, pi, res_ds])
                
    p2 = Pool(cp)
    res2_pool = p2.map(f2, x2_set)
    # --> tf_list, res_tfs
                
    for ih, h in enumerate(sample):
        fig8, axes8 = plt.subplots(3, 3, figsize=(11,11), tight_layout=True)
        for i in range(3):
            for j in range(3):
                axes8[i,j].set_xlim(1400,4600)
                axes8[i,j].set_ylim(0.85,9.15)
                axes8[i,j].set_xticks(range(1500,4501,1500))
                axes8[i,j].set_yticks(range(1,10,1))
                if i < 2:
                    axes8[i,j].tick_params(labelbottom=False)
                if j > 0:
                    axes8[i,j].tick_params(labelleft=False)
        
        for i, fit in enumerate(('A', 'B', 'C')):
            res_ds = res1_pool[3*ih+i]
            for j, pi in enumerate((50, 70, 90)):
                tf_list, res_tfs = res2_pool[9*ih+3*i+j]
                
                for k in range(len(x_test)):
                    for l in range(len(t_test)):
                        axes8[i,j].scatter(x_test[k], t_test[l], s=30, c=c_list[tf_list[k][l]])
                
                res_ds_zoom = scipy.ndimage.zoom(res_ds, 2)
                cpp1_z = scipy.ndimage.zoom(cpp1, 2)
                cpp2_z = scipy.ndimage.zoom(cpp2, 2)
                axes8[i,j].contour(cpp1_z, cpp2_z, res_ds_zoom, levels=[pi], colors='k')
        plt.show()