#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source code attached for the manuscript:

"Determination and validation of a design space for mesenchymal stem cell cultivation processes using prediction intervals" by
Keita Hirono, Yusuke Hayashi, Isuru A. Udugama, Mohamed Rami Gaddem, Kenjiro Tanaka, Yuto Takemoto, Ryuji Kato, Masahiro Kino-oka, Hirokazu Sugiyama

Last saved on Jan 23 2025


Lines     : Functions                                       (Corresponding section)
300 - 328 : Parameter estimation via Exp 1                  (Supplementary Result)
330 - 409 : Conventional DS determination and validation    (Supplementary Result)
425 - 435 : Prediction before parameter re-estimation       (Supplementary Result)
440 - 469 : Prediction interval calculation                 ("Prediction interval of the maximum specific growth rate and limits of growth prediction")
475 - 486 : Parameter reestimation                          ("Re-estimation of the maximum specific growth rate and model validation")
490 - 501 : Model validation                                ("Re-estimation of the maximum specific growth rate and model validation")
505 - 534 : Upper and lower limits of prediction            ("Prediction interval of the maximum specific growth rate and limits of growth prediction"))
540 - 571 : Stochastic simulation                           ("Prediction interval of the maximum specific growth rate and limits of growth prediction"))
575 - 641 : DS determination                                ("Determination and validation of the design space")
650 - 680 : DS validation                                   ("Determination and validation of the design space")

Created by Keita Hirono/Sugiyama-Badr Lab/The University of Tokyo
"""






import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy.optimize as optimize
from scipy.integrate import solve_ivp
import statistics
from matplotlib import cm
from matplotlib.colors import ListedColormap

def simu_grwoth(var, k, para, dv):
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


def calc_ode(t, var, k, para):
    X, GLC, LAC = var
    u_max = k[0]
    x_max, epsilon, t_lag, K_sd, q_glc, q_lac, S, V = para
    
    if u_max < 0:
       u = 0
       dXdt = 0
       dGlcdt = 0
       dLacdt = 0
    else:
        if t <= t_lag:
            u = 0
        else:
            f_con = 1 - X / x_max
            f_sd = 1 - K_sd * epsilon
            u = u_max * f_con * f_sd
            
        dXdt = u * X
        dGlcdt = - q_glc * X * S / V
        dLacdt = q_lac * X * S / V
         
    return [dXdt, dGlcdt, dLacdt]


def calc_nrmse(df_F, mse_list, S=1):
    mse = statistics.mean(mse_list)
    rmse = np.sqrt(mse)
    nrmse = rmse / ((df_F.max().max() - df_F.min().min()) / S) * 100
    return nrmse

def calc_mse(actual, predicted):
    residual = actual - predicted
    mse = np.mean(np.square(residual))
    return mse


def calc_error(k, df_F, df_IN, para, figs, graph=True, errorbar=True):
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
        y = simu_grwoth(var, k, para, dv)
        
        pred_list_1 = y[0][::60]
        t_true = df_F['time'][1:]
        X_true = df_F[sample][1:] / S
        
        mse_batch = []
        actual = np.array(X_true, dtype=float)
        predicted = np.array(pred_list_1, dtype=float)
        mse = calc_mse(actual, predicted)
        mse_batch.append(mse)
        mse_list.append(sum(mse_batch))
        
        if graph:
            nrmse_ave = plot_nrmse(df_F, mse_batch, nrmse_ave, n_sample, sample, t_true, X_true, y, S)
            
        ss += np.sum(np.square(actual - predicted))
    error = calc_nrmse(df_F, mse_list, S)
    
    if errorbar:
        #MEASUREMENT
        y_mean = df_F.iloc[:,1:].mean(axis=1)
        y_std = df_F.iloc[:,1:].std(axis=1)
        
        #FITTING CURVE/PREDICTION
        alpha, epsilon, x_max = df_IN.mean()
        para[0:2] = x_max, epsilon
        X0_i = df_F.iloc[0,1:].mean() * alpha
        var = [X0_i, 5, 2]
        y = simu_grwoth(var, k, para, dv)
        
        fig, ax = plt.subplots(figsize=(5,5))
        set_axplt(ax, dv['day_passage'])             
        ax.errorbar(df_F.loc[1:,['time']] / 24, y_mean[1:], yerr=y_std[1:], capsize=3, fmt='o', markersize=4, ecolor='k', markeredgecolor='k', color='w')
        ax.plot([y[-1][i] / 24 for i in range(len(y[-1]))], [y[0][i] * S for i in range(len(y[0]))], c=figs['color'], linestyle=figs['style'])
        ax.text(1, 12e+4, 'NRMSE = {}%'.format(round(error, 2)))
        plt.show()
        
    return ss


def plot_nrmse(df, mse_batch, nrmse_ave, n_ave, sample, t_true, X_true, y, S=1):
    nrmse_batch = [round(np.sqrt(h) / ((df.max().max() - df.min().min()) / S) * 100, 2) for h in mse_batch]
    nrmse_ave = [round(nrmse_ave[i] + nrmse_batch[i] / n_ave, 2) for i in range(len(nrmse_batch))]
    print(nrmse_batch, nrmse_ave)
    
    fig, ax = plt.subplots(figsize=(5,5))
    set_axplt(ax, dv['day_passage'])
    ax.set_title(sample)
    ax.scatter(t_true / 24, X_true * S, s=20, edgecolors='k', c='w')
    ax.plot([y[-1][i] / 24 for i in range(len(y[-1]))], [y[0][i] * S for i in range(len(y[0]))], c=figs['color'], linestyle=figs['style'])
    ax.text(1, 12e+4, 'NRMSE = {}%'.format(str(nrmse_batch[0])))
    plt.show()
    
    return nrmse_ave


def calc_metrics(x_test, t_test, rec_obs, cpp1, cpp2, res_ds, pi):
    for i in range(len(x_test)):
        for j in range(len(t_test)):
            obs = rec_obs[j,i]
            s = list(cpp1).index(x_test[i])
            t = list(cpp2).index(t_test[j])
            
            cal = res_ds[t,s]
            
            if pi <= obs and cal < pi:tf = 0 #fn
            if pi <= obs and pi <= cal:tf = 1 #tp
            if obs < pi and cal < pi:tf = 2 #tn
            if obs < pi and pi <= cal:tf = 3 #fp
            rec_tf[j,i] = tf
            
    fn = np.sum(rec_tf == 0)
    tp = np.sum(rec_tf == 1)
    tn = np.sum(rec_tf == 2)
    fp = np.sum(rec_tf == 3)
    res_tf = [tp, fp, tn, fn] 
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp)    
    lhr = sensitivity / (1 - specificity)
    return lhr, res_tf


def set_axds(ax):
    ax.set_xlabel('Seeding density [cells cm$^{-2}$]')
    ax.set_ylabel('Harvesting time [day]')
    ax.set_ylim(4,9)
    ax.set_xticks(np.arange(1500,4501,1500))
    ax.set_yticks(np.arange(4,9.1,0.5))
    
    
def set_axplt(ax, t_end):
    ax.set_xlabel('Time [day]'); ax.set_ylabel('Adhesion cell number [cells]')
    ax.set_ylim(0,14e+4)
    ax.set_xticks(np.arange(1,t_end+0.1,1))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci',  axis='y', scilimits=(0,0))
    

if __name__ == '__main__':
    seed = 1234
    rng = np.random.default_rng(seed)
    
    fit = 'A'
    pi = 90
    print(fit, pi)
    n_ds = 10
    
    GNBU_R = cm.get_cmap('GnBu_r', 201)
    NEWCLRS = GNBU_R(np.linspace(0, 1, 201))
    WHITE = np.array([1, 1, 1, 1])
    NEWCLRS[2*pi:, :] = WHITE
    NEWCMP = ListedColormap(NEWCLRS)
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.subplot.bottom'] = 0.25
    plt.rcParams['figure.subplot.left'] = 0.25
    
    ##FIXED DESIGN FOR EXP1
    S = 1.53 * 1.53       
    V = 0.2 * S / 1000  
    MC_in = [5, 2]
    
    ##FIXED PARAMS
    Q_glc = 1.078e-10       # mmol/(cell h)
    Q_lac = Q_glc * 2       # mmol/(cell h)
    K_sd = 24.7
    
    PARA = [0, 0, 0, K_sd, Q_glc, Q_lac, S, V]
    
    ##FIXED OPERATION
    Day_passage = 8
    MC_period = 3
    R_mc = [0.5 for _ in range((Day_passage-1) // MC_period)]
    dv = dict(mc_period=MC_period, day_passage=Day_passage, r_mc=R_mc, mc_in=MC_in)
      
    ##DATASET
    df0 = pd.read_csv('csv/Exp1_measured_data.csv')
    df0_IN = pd.read_csv('csv/Exp1_calculated_parameters.csv', index_col=0)
    
    
    
    ###Main
    #PARAMETER ESTIMATION VIA EXP1 [Fig S1a]
    u_max = 4.e-2           # 1/h
    k0 = [u_max]
    n_sample = df0.shape[1] - 1
    k_opt_list = []
    #PARAMETER ESTIMATION FOR EVERY SAMPLE
    for i in range(n_sample):
        figs = dict(name='fig1a_F_{}'.format(i+1), style='--', color='k')
        
        n0 = n_sample
        n_sample = 1
        
        df_fit_i = df0.iloc[:,[0,i+1]]
        df_in_i = pd.DataFrame([df0_IN.iloc[i,:]])
        
        graph = False; errorbar = False
        mi_nrmse = optimize.minimize(calc_error, k0, args=(df_fit_i, df_in_i, PARA, figs, graph, errorbar),
                                        method='Nelder-Mead', 
                                        options={'maxiter':5000})
        print(mi_nrmse)
        k_opt = mi_nrmse.x
        k_opt_list.append(k_opt[0])
        nrmse_F2 = calc_error(k_opt, df_fit_i, df_in_i, PARA, figs, graph)
        
        n_sample = n0
        
    mean = statistics.mean(k_opt_list)
    figs = dict(name='fig1a_F', style='--', color='k')
    nrmse_F2 = calc_error([mean], df0, df0_IN, PARA, figs, graph)
        
    #CONVENTIONAL DS DETERMINATION & VALIDATION [Fig S1b,c,d]
    k_opt = [4.023e-02]
    
    CQA1 = 5e+4
    CQA2 = 0.8
    n_cpp1 = 9
    n_cpp2 = 33
    
    cpp1 = np.linspace(1500,4500,n_cpp1)
    cpp2 = np.linspace(9,1,n_cpp2)
    
    dv['day_passage'] = 9
    
    res_ds = np.zeros((len(cpp2), len(cpp1)))
    
    df_all = df0_IN
    dist_xmax = rng.choice(df_all['xmax'], n_ds)
    dist_eps = rng.choice(df_all['epsilon'], n_ds)
    dist_alp = rng.choice(df_all['alpha'], n_ds)
        
    for s in range(len(cpp1)):
        x_seed = cpp1[s]
    
        if x_seed % 3000 == 0:
            fig1, ax1 = plt.subplots(figsize=(5,5))
            set_axplt(ax1, Day_passage)  
        
        for i in range(n_ds):
            x_max = dist_xmax[i]
            epsilon = dist_eps[i]
            alpha = dist_alp[i]
            
            X0_i = x_seed * alpha
            para = [x_max, epsilon, 0, K_sd, Q_glc, Q_lac, S, V]
            var = [X0_i, 5, 2]
            
            y = simu_grwoth(var, k_opt, para, dv)
            if x_seed % 3000 == 0:
                ax1.plot([y[-1][i] / 24 for i in range(len(y[-1]))], [y[0][i] * S for i in range(len(y[0]))], c='k', alpha=.05)
            
            for t in range(len(cpp2)):
                if cpp2[t] >= 2 or (x_seed in (1500, 3000, 4500)):
                    
                    hr = cpp2[t] * 24
                    ind = [round(t, 1) for t in y[-1]].index(hr)
                    
                    y1 = y[0][ind] * S
                    y2 = y[0][ind] / x_max
                    if CQA1 <= y1 and y2 < CQA2:
                        res_ds[t,s] += 100 / n_ds
                                  
    fig, ax = plt.subplots(figsize=(6,5))
    set_axds(ax)
    ax.set_ylim(1,9)
    ax.set_yticks(np.arange(1,9.1,1))
    ds2 = ax.contour(cpp1, cpp2, res_ds, levels=[pi], colors='k')
    ds = ax.contourf(cpp1, cpp2, res_ds, levels=range(0,101,10), cmap=NEWCMP)
    cbar = fig.colorbar(ds, ticks=range(0,101,10), label='Probability [%]')
    cbar.add_lines(ds2)
    plt.show()
    
    x_test = [1500, 3000, 4500]
    t_test = np.loadtxt('csv/Exp2_measured_probability_for_DS_validation.csv', delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[0])
    rec_obs = np.loadtxt('csv/Exp2_measured_probability_for_DS_validation.csv', delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[1,2,3])
    rec_tf = np.zeros(rec_obs.shape)
    
    lhr, res_tf = calc_metrics(x_test, t_test, rec_obs, cpp1, cpp2, res_ds, pi)
    print('TP, FP, TN, FN')
    print(res_tf)
    
    fig, ax = plt.subplots(figsize=(5,5))
    set_axds(ax)
    ax.set_xlim(1400,4600)
    ax.set_ylim(.85,9.15)
    ax.set_yticks(np.arange(1,9.1,1))
    ax.contour(cpp1, cpp2, res_ds, levels=[pi], colors='k')
    for i in range(len(x_test)):
        for j in range(len(t_test)):
            cs = ax.scatter(x_test[i], t_test[j], s=30, c=['orange','g','b','r'][int(rec_tf[j,i])])
    plt.show()
    
    
    
    
    
    ##FIXED OPERATION FOR EXP2
    Day_passage = 9
    MC_period = 9 
    R_mc = [0.5 for _ in range((Day_passage-1) // MC_period)]
    dv = dict(mc_period=MC_period, day_passage=Day_passage, r_mc=R_mc, mc_in=MC_in)
    
    ##DATASET
    df = pd.read_csv('csv/Exp2_measured_data.csv')
    df_IN = pd.read_csv('csv/Exp2_calculated_parameters.csv', index_col=0)
    
    #PREDICTION BEFORE PARAMETER RE-ESTIMATION [Fig S4]
    k_opt = [4.023e-02]
    for i in range(3):
        figs = dict(name='fig3a_V', style='-', color='k')
        df_val_i = pd.concat([df['time'], df.filter(like=str(1500*(i+1)), axis=1)], axis=1)
        
        n_sample = df_val_i.shape[1] - 1
        df0_in_i = df_IN.filter(like=str(1500*(i+1)), axis=0)
        df0_in_i.iloc[:,:] = df0_IN.mean()
        
        graph = False
        nrmse_V1 = calc_error(k_opt, df_val_i, df0_in_i, PARA, figs, graph)
       
    
                    
    #PREDICTION INTERVAL CALCULATION
    df_fit = pd.concat([df['time'], df.filter(like=fit, axis=1)], axis=1)
    n_sample = df_fit.shape[1] - 1
    df_in = df_IN.filter(like=fit, axis=0)
    
    k2_opt_list = []
    
    #PARAMETER ESTIMATION FOR EVERY SAMPLE
    for i in range(n_sample):
        figs = dict(name='stable1/stable1_{}_fit_{}'.format(fit, i+1), style='--', color='r')
        
        n0 = n_sample
        n_sample = 1
        
        df_fit_PI = df_fit.iloc[:,[0,i+1]]
        df_in_PI = pd.DataFrame([df_in.iloc[i,:]])
        
        graph = False; errorbar = False
        mi_nrmse = optimize.minimize(calc_error, k_opt, args=(df_fit_PI, df_in_PI, PARA, figs, graph, errorbar),
                                        method='Nelder-Mead', 
                                        options={'maxiter':5000})
        k2_opt = mi_nrmse.x
        k2_opt_list.append(k2_opt[0])
        nrmse_F2 = calc_error(k2_opt, df_fit_PI, df_in_PI, PARA, figs, graph)
        
        n_sample = n0
    
    mean = statistics.mean(k2_opt_list)
    std = statistics.stdev(k2_opt_list)
    ta = 2.110 #t value of dof=18-1 for 95% confidence interval
    pi_lb, pi_ub = mean - ta * std * np.sqrt(1 + 1 / 18), mean + ta * std * np.sqrt(1 + 1 / 18)
                   
                                
                                                  
            
    #PARAMETER REESTIMATION [Figs 2a and S2]
    k2_opt = {'A': [2.755e-2], 'B': [2.738e-02], 'C': [2.718e-02]} 
    df_fit = df.filter(like=fit, axis=1)
    df_in = df_IN.filter(like=fit, axis=0)
        
    for i in range(3):
        figs = dict(name='fig3a_{}_fit'.format(fit), style='--', color='r')
        df_in_i = df_in.filter(like=str(1500*(i+1)), axis=0) 
        df_fit_i = pd.concat([df['time'], df_fit.filter(like=str(1500*(i+1)), axis=1)], axis=1) 
        n_sample = df_fit_i.shape[1] - 1
        
        graph = False
        nrmse_F2 = calc_error(k2_opt[fit], df_fit_i, df_in_i, PARA, figs, graph)
            
            
    #MODEL VALIDATION [Fig 2b and S2]
    df_val = df.loc[:, ~df.columns.str.contains(fit)]
    df_in = df_IN.loc[~df_IN.index.str.contains(fit)]
    df_all = pd.concat([df0_IN, df_IN.filter(like=fit, axis=0)], axis=0)
    df_in.iloc[:,:] = df_all.mean()
        
    for i in range(3):
        figs = dict(name='fig3b_{}_val'.format(fit), style='-', color='tab:orange')
        df_val_i = pd.concat([df['time'], df_val.filter(like=str(1500*(i+1)), axis=1)], axis=1)
        n_sample = df_val_i.shape[1] - 1
        
        graph = False
        nrmse_V2 = calc_error(k2_opt[fit], df_val_i, df_in, PARA, figs, graph)
            
    
    #UPPER AND LOWER LIMITS OF PREDICTION [Figs 3a and S2]
    pi_dic = {'A': [[2.440e-2], [3.070e-2]], 'B': [[2.444e-2], [3.033e-2]], 'C': [[2.236e-2], [3.200e-2]]}
    df_val = df.loc[:, ~df.columns.str.contains(fit)]
    df_in = df_IN.loc[~df_IN.index.str.contains(fit)]
    df_all = pd.concat([df0_IN, df_IN.filter(like=fit, axis=0)], axis=0)
    df_in.iloc[:,:] = df_all.mean()
    
    for i in range(3):
        df_val_i = pd.concat([df['time'], df_val.filter(like=str(1500*(i+1)), axis=1)], axis=1)
        x_seed = df_val_i.iloc[0, 1]
        
        fig, ax = plt.subplots(figsize=(5,5))
        set_axplt(ax, Day_passage)
        
        y_mean = df_val_i.iloc[:,1:].mean(axis=1)
        y_std = df_val_i.iloc[:,1:].std(axis=1)
        ax.errorbar(df_val_i.loc[1:,['time']] / 24, y_mean[1:], yerr=y_std[1:], capsize=3, fmt='o', markersize=4, ecolor='k', markeredgecolor='k', color='w')
        
        x_max = df_in['xmax'][0]
        epsilon = df_in['epsilon'][0]
        alpha = df_in['alpha'][0]
        X0_i = x_seed * alpha
        para = [x_max, epsilon, 0, K_sd, Q_glc, Q_lac, S, V]
        var = [X0_i, 5, 2]
        
        y_lb = simu_grwoth(var, pi_dic[fit][0], para, dv)
        y_ub = simu_grwoth(var, pi_dic[fit][1], para, dv)
        ax.plot([y_lb[-1][i] / 24 for i in range(len(y_lb[-1]))], [y_lb[0][i] * S for i in range(len(y_lb[0]))], c='b', linestyle='-')
        ax.plot([y_ub[-1][i] / 24 for i in range(len(y_ub[-1]))], [y_ub[0][i] * S for i in range(len(y_ub[0]))], c='g', linestyle='-')
        
        plt.show()
            
    
    
    
    #STOCHASTIC SIMULATION [Fig 3b]
    cpp1 = np.linspace(1500, 4500, 3)
    df_all = pd.concat([df0_IN, df_IN.filter(like=fit, axis=0)], axis=0)
    dist_xmax = rng.choice(df_all['xmax'], n_ds)
    dist_eps = rng.choice(df_all['epsilon'], n_ds)
    dist_alp = rng.choice(df_all['alpha'], n_ds)
        
    for s in range(len(cpp1)):
        x_seed = cpp1[s]
    
        fig1, ax1 = plt.subplots(figsize=(5,5))
        fig2, ax2 = plt.subplots(figsize=(5,5))
        set_axplt(ax1, Day_passage); set_axplt(ax2, Day_passage)        
                
        for i in range(n_ds):
            x_max = dist_xmax[i]
            epsilon = dist_eps[i]
            alpha = dist_alp[i]
            
            X0_i = x_seed * alpha
            para = [x_max, epsilon, 0, K_sd, Q_glc, Q_lac, S, V]
            var = [X0_i, 5, 2]
            
            #average
            y = simu_grwoth(var, k2_opt[fit], para, dv)
            ax1.plot([y[-1][i] / 24 for i in range(len(y[-1]))], [y[0][i] * S for i in range(len(y[0]))], c='k', alpha=.05)
            
            #PI
            y2_lb = simu_grwoth(var, pi_dic[fit][0], para, dv)
            y2_ub = simu_grwoth(var, pi_dic[fit][1], para, dv)
            ax2.plot([y2_lb[-1][i] / 24 for i in range(len(y2_lb[-1]))], [y2_lb[0][i] * S for i in range(len(y2_lb[0]))], c='b', alpha=.05)
            ax2.plot([y2_ub[-1][i] / 24 for i in range(len(y2_ub[-1]))], [y2_ub[0][i] * S for i in range(len(y2_ub[0]))], c='g', alpha=.05)
        plt.show()
        
    
    #DS DETERMINATION [Figs 4a, 5a, 5c, and S3]
    CQA1 = 5e+4
    CQA2 = 0.8
    n_cpp1 = 9
    n_cpp2 = 33
    
    cpp1 = np.linspace(1500,4500,n_cpp1)
    cpp2 = np.linspace(9,1,n_cpp2)
    
    res_ds = np.zeros((len(cpp2), len(cpp1)))
    res2_ds = np.zeros((len(cpp2), len(cpp1)))
    
    df_all = pd.concat([df0_IN, df_IN.filter(like=fit, axis=0)], axis=0)
    dist_xmax = rng.choice(df_all['xmax'], n_ds)
    dist_eps = rng.choice(df_all['epsilon'], n_ds)
    dist_alp = rng.choice(df_all['alpha'], n_ds)
        
    for s in range(len(cpp1)):
        x_seed = cpp1[s]
    
        for i in range(n_ds):
            x_max = dist_xmax[i]
            epsilon = dist_eps[i]
            alpha = dist_alp[i]
            
            X0_i = x_seed * alpha
            para = [x_max, epsilon, 0, K_sd, Q_glc, Q_lac, S, V]
            var = [X0_i, 5, 2]
            
            y = simu_grwoth(var, k2_opt[fit], para, dv)
            y_lb = simu_grwoth(var, pi_dic[fit][0], para, dv)
            y_ub = simu_grwoth(var, pi_dic[fit][1], para, dv)
            
            for t in range(len(cpp2)):
                if cpp2[t] >= 4 or (x_seed in (1500, 3000, 4500)):
                    
                    hr = cpp2[t] * 24
                    ind = [round(t, 1) for t in y_lb[-1]].index(hr)
                    
                    y1 = y[0][ind] * S
                    y2 = y[0][ind] / x_max
                    if CQA1 <= y1 and y2 < CQA2:
                        res_ds[t,s] += 100 / n_ds
                                    
                    y1_lb, y1_ub = y_lb[0][ind] * S, y_ub[0][ind] * S
                    y2_lb, y2_ub = y_lb[0][ind] / x_max, y_ub[0][ind] / x_max
                    if CQA1 <= min(y1_lb, y1_ub) and max(y2_lb, y2_ub) < CQA2:
                        res2_ds[t,s] += 100 / n_ds
     
    fig, ax = plt.subplots(figsize=(6,5))
    set_axds(ax)
    ax.set_ylim(1,9)
    ax.set_yticks(np.arange(1,9.1,1))
    ds2 = ax.contour(cpp1, cpp2, res_ds, levels=[pi], colors='k')
    ds = ax.contourf(cpp1, cpp2, res_ds, levels=np.linspace(0,100,11), cmap=NEWCMP)
    cbar = fig.colorbar(ds, ticks=np.linspace(0,100,11), label='Probability [%]')
    cbar.add_lines(ds2)
    plt.show()
        
    fig, ax = plt.subplots(figsize=(6,5))
    set_axds(ax)
    ax.set_ylim(1,9)
    ax.set_yticks(np.arange(1,9.1,1))
    ds2 = ax.contour(cpp1, cpp2, res2_ds, levels=[pi], colors='k')
    ds = ax.contourf(cpp1, cpp2, res2_ds, levels=np.linspace(0,100,11), cmap=NEWCMP)
    cbar = fig.colorbar(ds, ticks=np.linspace(0,100,11), label='Probability [%]')
    cbar.add_lines(ds2)
    plt.show()
        
    
    
    
    
    
    
    #DS VALIDATION [Figs 4b, 5b, 5d, 6, and S4]
    rec_obs = np.loadtxt('csv/Exp2{}_measured_probability_for_DS_validation.csv'.format(fit), delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[1,2,3])
    rec_tf = np.zeros(rec_obs.shape)
    
    x_test = [1500, 3000, 4500]
    t_test = np.loadtxt('csv/Exp2{}_measured_probability_for_DS_validation.csv'.format(fit), delimiter=',', skiprows=1, max_rows=n_cpp2, usecols=[0])
    
    for h, res in enumerate([res_ds, res2_ds]):
        lhr, res_tf = calc_metrics(x_test, t_test, rec_obs, cpp1, cpp2, res, pi)
        print('TP, FP, TN, FN')
        print(res_tf)
        
        tf_list = []
        for j in range(3):
            tfs = []
            for obs, cal in zip(rec_obs[:,j], res[:,4*j]):
                if pi <= obs and cal < pi:tfs.append('FN')
                if pi <= obs and pi <= cal:tfs.append('TP')
                if obs < pi and cal < pi:tfs.append('TN')
                if obs < pi and pi <= cal:tfs.append('FP')
            tf_list.append(tfs)
        
        fig, ax = plt.subplots(figsize=(5,5))
        set_axds(ax)
        ax.set_xlim(1400,4600)
        ax.set_ylim(0.85,9.15)
        ax.set_yticks(np.arange(1,9.1,1))
        ds2 = ax.contour(cpp1, cpp2, res, levels=[pi], colors='k')
        for i in range(len(x_test)):
            for j in range(len(t_test)):
                cs = ax.scatter(x_test[i], t_test[j], s=30,c=['orange','g','b','r'][int(rec_tf[j,i])])
        plt.show()
        
    
    
    