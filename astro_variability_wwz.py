#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
Created on Sun Apr 10 13:57:39 2022

@author: akshayghosh

This file includes functions to study variability of lightcurves such as calculating
the power spectral density (PSD) or continuous wavelet transform (CWT)

LIST OF FUNCTIONS AND INPUTS:
    
    av.generate_lightcurve(b = 2.0, N = np.power(2,15), plot = False)
    av.generate_long_lightcurve(sections = 1000, data_points = np.power(2,13),plot = False, b = 2.0)
    av.E2013_LC(sections = 1, data_points = np.power(2,7),plot = True, power_law = 2.0, input_pdf = 'lognormal', poisson_noise = True, num_iterations = 80, scale = 1, noise = 1, t_stop = False)
    av.calc_cwt(x,t, plot = True, LC_name = False, num_levels = 20, num_widths = 20, wavelet_type = 'morl', freq_bins = 'log',freq_min = 'auto', freq_max = 'auto', CI = False, COI = False)
    av.calc_psd(x,t, plot = True, LC_name = False, normalize = True)
    av.calc_pdf(x,t,num_bins = 20, LC_name = False, plot = False)
    av.frequency2scale(f,t,wavelet_type = 'morl')
    av.generate_poisson_noise(x,t, plot = False, noise_level = 1)
    av.read_flc(f, plot = False, trunc_non_zero = True)
    av.plot_lightcurve(x,t,LC_name = False, bin_info = True, time_unit = 's')
    av.plot_psd(P,ν,LC_name = False)
    av.plot_cwt(T = 0,t = 0,f = 0,LC_name = False,levels = 20)
    av.plot_pdf(x,t,num_bins = 20, LC_name = False)
    av.adjust_bin_width(x,t,new_bin_width)
    av.calc_global_cwt(T,t,freq, plot = True, normalize = True)
    av.add_periodic_gaps(x,t,num_gaps = 3,gap_length = 0.7, random_gap_length = 0, plot = True, random_gaps = False,gap_type = 'zero')
    av.python_to_matlab(x,t,folder_path = '/Users/akshayghosh/wavelet_analysis/matlab_files/', LC_name = False, timestamp = True)
    av.generate_simulated_lc_ensemble(FILENAME_in,LC_NAME_out, b = 2.0, K = 1000)
"""

# IMPORTS

# import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import pi
#from matplotlib.ticker import MultipleLocator, LogLocator

# import seaborn as sns
# # set seaborn style
# sns.set_style("white")
# sns.set()
# import matplotlib.colors as colors
# import matplotlib.cbook as cbook
from matplotlib import ticker, cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullLocator)
#import scipy.fftpack
from scipy import signal
from scipy.optimize import curve_fit
# from astropy.io import fits
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import chisquare
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from scipy.stats import cauchy
from scipy.stats import linregress
from scipy.interpolate import interp1d

from scipy.io import savemat
from scipy.io import loadmat

from datetime import datetime
import time

from sklearn.metrics import auc

import matplotlib.gridspec as gridspec

from astropy.timeseries import LombScargle
import pyleoclim as pyleo

from astropy.io import fits



# import stingray
# import warnings
# warnings.filterwarnings('ignore')
# from stingray import Lightcurve

# from mpl_toolkits.mplot3d import Axes3D

import sys # from derek's structure function code
from sklearn.linear_model import LinearRegression # from derek's structure function code
import concurrent.futures as cf

from tqdm import tqdm # progress bar

import sort_by_ranking_FORTRAN # use a fortran subroutine to sort by ranking for E2013 lc method



######################################################################################################################################

# DATA

lightcurves = ['/Users/akshayghosh/wavelet_analysis/lightcurves/LC_0464_01.flc',
               '/Users/akshayghosh/wavelet_analysis/lightcurves/LC_1027_01.flc',
               '/Users/akshayghosh/wavelet_analysis/lightcurves/LC_2768_01.flc',
               '/Users/akshayghosh/wavelet_analysis/lightcurves/LC_2769_01.flc',
               '/Users/akshayghosh/wavelet_analysis/lightcurves/LC_3680_01.flc',
               '/Users/akshayghosh/wavelet_analysis/lightcurves/LC_3681_01.flc']

######################################################################################################################################

# FUNCTIONS

def generate_lightcurve(b = 2.0, N = np.power(2,15), plot = False,t_start = 0,t_stop = 10000,qpo = False,cauchy_centers = None,cauchy_amps = None,cauchy_scales = None, custom_psd = False):
    
    # initalize fourier transform vector
    FT = np.zeros(N, dtype = 'cdouble')
    
    # if custom_psd is set to false, make one here with the power law b
    if type(custom_psd) == bool:
        # initialize fourier freqs w, w = 2πf, so need to use f = w/2π
        w_start = 1 / (t_stop - t_start)
        w_stop = N / (t_stop - t_start)
        w = np.linspace(w_start,w_stop,N) # / (2*pi)
        # w = np.linspace(w_start,2*w_stop,2*N)
        
        S = np.power(w,-b) # add lorentzian to this
        # print('NORMALZING IN TK95')
        S = S / auc(w,S)
        
        # print('pl = {}'.format(b))
        # S *= 1e-9
        
        # plt.figure()
        # plt.plot(w,S)
        # plt.title('INPUT PSD');plt.xlabel('Freq');plt.ylabel('Amp');plt.xscale('log');plt.yscale('log')
        # plt.show()
        
        # qpo = False
        if qpo:
            # print('adding qpo')
            num_lorentzians = len(cauchy_centers)
            cauchy_signal = np.zeros(N)
        
            for sig in range(num_lorentzians):
                cauchy_center = cauchy_centers[sig]
                cauchy_amp = cauchy_amps[sig]
                cauchy_scale = cauchy_scales[sig]
                cauchy_signal += np.power(10.0,cauchy_amp) * cauchy.pdf(w, loc = cauchy_center,scale = np.power(10.0,cauchy_scale)) 
            
            S = S + cauchy_signal
            # S = cauchy_signal
            # S *= np.power(10.0,-5.0) # scale PSD to be in realistic range
        # plt.figure()
        # plt.plot(w,S)
        # plt.title('INPUT PSD');plt.xlabel('Freq');plt.ylabel('Amp');plt.xscale('log');plt.yscale('log')
        # plt.show()
       
    # for a custom psd input a vector that looks like [freq_vector, psd_vector]
    else:
        w = custom_psd[0]
        S = custom_psd[1]
        # if True:
        #     plt.figure()
        #     plt.plot(w,S)
        #     plt.title('INPUT PSD');plt.xlabel('Freq');plt.ylabel('Amp');plt.xscale('log');plt.yscale('log')
        #     plt.show()
    # S *= 1e2 / S[0] # scale PSD to be in realistic range
    
    S_sqrt = np.power(S,0.5)
    
    
    # multiply each fourier freq by random gaussian numbers for real and imag part
    random_μ = 0
    random_σ = 1#100#10000 # used to be 10, increased to attempt to have less apparent signals
    random_num_realpart = np.random.normal(random_μ,random_σ, size = N)
    random_num_imagpart = np.random.normal(random_μ,random_σ, size = N)
    
    FT = S_sqrt*random_num_realpart + 1j*S_sqrt*random_num_imagpart
    
    # inverse FT to get time series aka lightcurve x(t) (t is arb)
    # x = np.ffst.ifft(FT[:N//2])
    x = np.fft.ifft(FT)
    t = np.linspace(t_start,t_stop,N)
    
    x = np.real(x)
    x = x.astype(dtype = 'float64')
    
    x = match_limits(vec_to_scale = x,vec_to_match = [0,20]) # need to scale x to be in realistic range
    
    # tk95_plot = plot_lightcurve(x,t,LC_name = 'TK95 lightcurve scaled')
    
    # PLOT LIGHTCURVE
    if plot:
        plt.figure()
        plt.plot(t,x)
        plt.title('Simulated Lightcurve')
        plt.xlabel('Time (s)')
        plt.ylabel('Count Rate')
        plt.show()
    
    return(x,t)

######################################################################################################################################

def generate_long_lightcurve(sections = 1, data_points = np.power(2,13),plot = False, b = 2.0, t_start = 0, t_stop = 100000,
                             qpo = False,cauchy_centers = None,cauchy_amps = None,cauchy_scales = None,custom_psd = False):
    
    # t_start = 0 # for each section interval
    # t_stop = 10 # for each section interval
    # k = 1000 # sections to generate
    k = sections
    K = data_points
    # K = np.power(2,13) #int(t_stop/1) # number of data points per section
    
    x_grid = np.zeros(shape = (k,K), dtype = 'float64')
    
    for i in range(k):
        x_grid[i],t = generate_lightcurve(b,N = K,t_start = 0, t_stop = t_stop,qpo = qpo,cauchy_centers = cauchy_centers,cauchy_amps = cauchy_amps,cauchy_scales = cauchy_scales)
    
    x = np.concatenate(x_grid)
    t = np.linspace(t_start,t_stop*k,k*K)
    
    # CONVERT AMPLITUDE TO COUNTS
    if True:
        count_scale_factor = 10
        x = x - np.min(x)
        x = count_scale_factor * (x/np.max(x))
        
    # PLOT LIGHTCURVE
    if plot:
        plt.figure()
        plt.plot(t,x)
        plt.title('Simulated Lightcurve')
        plt.xlabel('Time (s)')
        plt.ylabel('Count Rate')
        plt.show()
        
    # PLOT HIST OF COUNTS
    if False:        
        plt.figure()
        plt.hist(x, bins = 200)
        plt.title('Distribution of Counts')
        plt.xlabel('Counts')
        plt.show()
    
    return(x,t)

######################################################################################################################################

def E2013_LC(sections = 1, data_points = np.power(2,7),plot = False, power_law = 2.0, input_pdf = 'lognormal', poisson_noise = True, num_iterations = 80,
             scale = False, noise = 1, t_stop = False,qpo = False,cauchy_centers = None,cauchy_amps = None,cauchy_scales = None,
             random_scaling = False,custom_psd = False,LC_name = False, non_uniform_sampling = False):
    
    # x_norm,t = generate_long_lightcurve(sections = 100, data_points = np.power(2,7) ,plot = False, b = power_law) # ii # note to pass power law β
    # x_norm,t = generate_long_lightcurve(sections, data_points ,plot = False, b = power_law, t_start = 0, t_stop = t_stop, qpo = qpo,
    #                                     cauchy_centers = cauchy_centers,cauchy_amps = cauchy_amps,cauchy_scales = cauchy_scales) # ii # note to pass power law β  
    
    x_norm,t = generate_lightcurve(b = power_law,N = data_points,t_start = 0, t_stop = t_stop,qpo = qpo,
                                    cauchy_centers = cauchy_centers,cauchy_amps = cauchy_amps,cauchy_scales = cauchy_scales,custom_psd = custom_psd)

    N = len(x_norm)
    
    DFT_norm_j = np.fft.fft(x_norm) # iii
    DFT_norm_j_real = np.real(DFT_norm_j)
    DFT_norm_j_imag = np.imag(DFT_norm_j)
    
    Δt = (np.max(t) - np.min(t))/N
    j = np.fft.fftfreq(n = N, d = Δt)
    
    A_norm_j = (1/float(N)) * np.sqrt( np.square(DFT_norm_j_real) +  np.square(DFT_norm_j_imag)) # iv
    φ_norm_j = np.angle(DFT_norm_j) # v
    P_norm = np.square(A_norm_j)
    
    
    convergence = False
    convergence_threshold = 1
    iteration_count = 0
    convergence_check_list = []
    
    # STEP 2:
    if type(input_pdf) == str:
        if input_pdf == 'wide gaussian':
            x_sim_1 = np.random.normal(loc = np.mean(x_norm), scale = 10*np.std(x_norm), size = N) # temp gaussian dist # i, ii
        elif input_pdf == 'uniform':
            x_sim_1 = 30*np.random.rand(N) # uniform distribution
        elif input_pdf == 'poisson':
            x_sim_1 = np.random.poisson(lam = 5, size = N)
        elif input_pdf == 'lognormal':
            μ_x = np.mean(x_norm)
            var_x = np.var(x_norm)
            lognormal_mean = np.log(np.square(μ_x) / np.sqrt(np.square(μ_x) + np.square(var_x)))
            lognormal_sigma = np.log(1 + np.square(var_x) / np.square(μ_x))
            x_sim_1 = np.random.lognormal(mean = lognormal_mean, sigma = lognormal_sigma, size = N) # lognormal distribution
            #plt.figure();plt.hist(x_sim_1,bins = 50);plt.xlabel('Counts/s');plt.show()
        else:
            # can also input a lightcurve as a csv file and use that PDF
            # print('using pdf from {}'.format(input_pdf))
            x_obs,t_obs,bg,x_err,bg_err,obs_title = csv_to_lcdata(input_pdf)
            PDF = calc_pdf(x_obs,t_obs,num_bins = 30, LC_name = False, plot = False) # PDF = [counts_obs,pdf_obs]
            # print('max counts: {}'.format(np.max(PDF[0])))
            x_sim_1 = np.random.choice(a = PDF[0], p = PDF[1], size = N)
    else:
        # for custom pdf, send array thats like input_pdf = (counts,probabilities)
        x_sim_1 = np.random.choice(a = input_pdf[0], p = input_pdf[1], size = N)
    
    # # x_sim_1 from distribution of real lightcurve:
    # x_obs,t_obs,LC_name = read_flc(f = lightcurves[1], plot = False)
    # x_obs_dist,bin_edges = np.histogram(a = x_obs,bins = len(x_obs))
    # x_obs_dist_normalized = x_obs_dist/np.sum(x_obs_dist)
    # x_sim_1 = np.random.choice(a = x_obs, p = x_obs_dist_normalized, size = N)
    
    # while convergence == False:
    t1 = time.time()
    for i in range(num_iterations):
        
        DFT_sim_1_j = np.fft.fft(x_sim_1)
        DFT_sim_1_j_real = np.real(DFT_sim_1_j)
        DFT_sim_1_j_imag = np.imag(DFT_sim_1_j)
        
        A_sim_1_j = (1/float(N)) * np.sqrt( np.square(DFT_sim_1_j_real) +  np.square(DFT_sim_1_j_imag))
        φ_sim_1_j = np.angle(DFT_sim_1_j)
        P_sim_1 = np.square(A_sim_1_j)
        
        
        # STEP 3:
        
        DFT_sim_adjust_1 = A_norm_j * np.exp(1j * φ_sim_1_j) # paper doesn't say to multiply by N, i just tried it
        x_sim_adjust_1 = np.fft.ifft(DFT_sim_adjust_1) # the amplitude is (or was?) too low by a factor of ~4
        x_sim_adjust_1 = np.real(x_sim_adjust_1)
        
        P1,f1 = calc_psd(x_sim_adjust_1, t, plot = False, LC_name = r'Simulated PSD from $A_{norm},\phi_{sim,1}$')
        
        # STEP 4:
        
        x_sim_2 = np.zeros(N) # initalize x_sim_2, replace q_th highest value with q_th highest value of x_sim_1
        
        '''
        Amplitude adjustment: A new time series is created from the values of xsim,1(t) ordered based on the ranking of xsim.adjust,1(t). 
        This means that the the highest value of xsim.adjust,1(t) = A is replaced by the highest value of xsim,1(t) = B, the second highest value 
        of xsim.adjust,1(t) is replaced by the second highest value of xsim,1(t) and so on.
        '''
        
        A = x_sim_adjust_1
        B = x_sim_1
        C = np.zeros(N,dtype = np.float32) # will later set this equal to x_sim_2, needs to be type float32 for the fortran func
        
        # A_idx = (rankdata(A) - 1).astype(int) # this is the sorting loop as 1 line
        # C = np.put(C,A_idx,B)
        
        # B_idx = (rankdata(B) - 1).astype(int) # this is the sorting loop as 1 line
        # C = np.put(C,B_idx,A)
        
        
        # SORT BY RANKING
        # C_test = B[np.argsort(A)] # this doesn't work but i wish it did
        
        sort_by_ranking_method = 'FORTRAN'
        
        # # TEST THE FORTRAN METHOD
        # C_f = np.zeros(N,dtype = np.float32)
        # sort_by_ranking_FORTRAN.sort_by_ranking_fortran(A, B, C_f, N)
        
        '''
        f2py -m sort_by_ranking_FORTRAN -c sort_by_ranking_FORTRAN.f90 # to set up a fortran subroutine to use in python!!
        '''
        
        if sort_by_ranking_method == 'PYTHON':
            for j in range(N):
                highest_value_A = np.max(A) # find highest value in A
                highest_value_A_idx = np.argmax(A) # find index of highest value in A, USE THIS INDEX
                highest_value_B = np.max(B) # find highest value in B, USE THIS VALUE
                highest_value_B_idx = np.argmax(B) # find index of highest value in B
                
                C[highest_value_A_idx] = highest_value_B # replace the highest value in A with the highest value in B
                # A[highest_value_A_idx] = 0 # set these to zero to find the next highest value and index
                # B[highest_value_B_idx] = 0
                A[highest_value_A_idx] = -1e6 # set these to zero to find the next highest value and index
                B[highest_value_B_idx] = -1e6
        # print('ended sorting')
        if sort_by_ranking_method == 'FORTRAN':
            # print('using fortran')
            sort_by_ranking_FORTRAN.sort_by_ranking_fortran(A, B, C, N)
        
        # C_test = B[np.argsort(A)]
        # C_test_results = C_test / C
        # print('SORT BY RANKING RESULTS:')
        # print(C_test_results)
        
        
        x_sim_2 = C # array to return and compare to x_sim_1 to check for convergence
        
        # STEP 5:
        x_sim_1 = x_sim_2 # convergence test was breaking the sim for some revs
        
        # # check for convergence # TEMP COMMENTED OUT START
        # convergence_check = np.mean(x_sim_2 - x_sim_1)
        # if convergence_check < convergence_threshold:
        #     convergence = True
        # else:
        #     x_sim_1 = x_sim_2
        #     #convergence_check_list.append(convergence_check) # commented this out to run better
        #     iteration_count += 1
        #     # print('iteration: {}'.format(iteration_count)) # TEMP COMMENTED OUT END
    
    # END FOR
    # C_test_results = C_test / C
    # print('SORT BY RANKING RESULTS:')
    # print(C_test)
    # print(C)
    # plt.figure();plt.plot(C_test,'b');plt.plot(C,'r');plt.show()
    
    
    # t2 = time.time()
    # delta_t = t2-t1
    # print(f'delta t for method = {sort_by_ranking_method}: {delta_t} s')
    
    if t_stop:
        t = np.linspace(0,t_stop,N)
    
    # START IF
    if type(scale) == str:
        if scale == 'max':
            # print('using max counts to scale')
            x_sim_1 = 1 * np.max(PDF[0]) * x_sim_1 / np.max(x_sim_1)
    
    elif type(scale) == bool:
        if scale:
            x_sim_1 = 1 * scale * x_sim_1 / np.max(x_sim_1) # set max value to scale counts
        # x_sim_1 = 10 * x_sim_1 / np.max(x_sim_1) # set max value to 10 counts
    # x = x_sim_1
    
    elif type(scale) == float or type(scale) == int:
        # print('scale = {}'.format(scale))
        # x_sim_1 = 1 * np.abs(scale) * x_sim_1 / np.max(x_sim_1) # scale by MAX
        inital_x_sim_1_avg = np.mean(x_sim_1) # scale by the input average
        x_sim_1 = (scale / inital_x_sim_1_avg) * x_sim_1
    # END IF
    
    if random_scaling:
        scale_factor = np.random.normal(loc = 10, scale = 5)
        x_sim_1 = scale_factor * x_sim_1/np.max(x_sim_1)
    
    # add poisson noise
    if poisson_noise:
        # plot_lightcurve(x_sim_1,t, LC_name = 'Without Poisson noise') # JUST FOR P NOISE FOR PRES
        x_sim_1 = generate_poisson_noise(x_sim_1, t, noise_level = noise)
        # plot_lightcurve(x_sim_1,t, LC_name = 'With Poisson noise') # JUST FOR P NOISE FOR PRES
        # x = x_pois
        # P_pois,f = calc_power_spectrum(x_pois, t,plot = True,LC_name = 'PSD final light curve with Poisson noise')

    if type(non_uniform_sampling) != bool:
        # print('interpolating to non-uniform t grid')
        x_sim_1 = interp1d(t, x_sim_1, kind='cubic')(non_uniform_sampling)
    
    
    if plot:
        if type(input_pdf) == str:
            if LC_name:
                plot_lightcurve(x_sim_1,t, LC_name = LC_name)
            else:
                plot_lightcurve(x_sim_1,t, LC_name = 'Simulated Lightcurve with input PDF: {}'.format(input_pdf))
        else:
            if LC_name:
                plot_lightcurve(x_sim_1,t, LC_name = LC_name)
            else:
                plot_lightcurve(x_sim_1,t, LC_name = 'Simulated Lightcurve')
        # P,f = calc_psd(x_sim_1, t,plot = True)
    

    return(x_sim_1,t)

######################################################################################################################################

def calc_cwt(x,t, plot = True, LC_name = False, num_levels = 20, num_widths = 20, wavelet_type = 'morl', freq_bins = 'log',freq_min = 'auto', freq_max = 'auto', CI = True,
             COI = True, COI_factor = 1,vmin_q = 0.1):
    
    # w_low = 2
    # w_high = 10000
    
    # DETECT FREQ RANGE TO CALC CWT
    if freq_min == 'auto' and freq_max == 'auto':
        P,f = calc_psd(x,t,plot = False)
        freq_min = f[1] # f[0] = 0 which is unphysical and also breaks the cwt calculation, so use f[1]
        freq_max = f[-3] # arbitrarly take 3rd last freq component as highest freq
    
    
    w_low = frequency2scale(float(freq_max),t)
    w_high = frequency2scale(float(freq_min),t)
    
    if freq_bins == 'linear' or freq_bins == None:
        widths = np.linspace(w_low,w_high,num_widths) # this is "scales"
    if freq_bins == 'log':
        widths = np.logspace(np.log10(w_low),np.log10(w_high),num_widths)
        # widths = np.logspace(np.log10(w_high),np.log10(w_low),num_widths) # reverse the vector to better represent frequencies
    S_P = (np.max(t) - np.min(t))/len(t) # sampling period
    
    # pad zeros on either side
    if False:
        k = np.log2(np.size(t)) # there are 2^k indexs in the LC
        pad_length = int((np.power(2,k + 1) - np.power(2,k)) / 2)
        pad = np.zeros(pad_length)
        x = np.concatenate([pad,x,pad])
        t_pad = np.linspace(0,1,len(x))
    
    # wavelet_type = 'mexh'
    wt,freqs = pywt.cwt(x, widths, wavelet_type, sampling_period = S_P, method = 'fft')
    # X,Y = np.meshgrid(t,widths)

    T = np.conj(wt)*wt # mod square
    # T = T/np.max(T) # set max value to 1
    
    # CALCULATE levels FOR CWT:
    if False:
        if freq_min < 0.001: # 0.001 seems like the min freq that doesn't break the level part
            level_min = -3
        else:
            level_min = np.log10(np.min(T))

    level_max = np.log10(np.max(T)) # because T_max is set to 1
    level_min = np.log10(np.min(T))
    # num_levels = 100 # resolution of contour plot
    levels_ = np.logspace(level_min,level_max,num_levels)
    # levels = np.linspace(np.min(T),np.max(T),num_levels)
    
    freq_var = pywt.scale2frequency(wavelet_type,widths)/S_P # convert scale to freq
    
    # freq_var = freqs
    
    if True:
        print('frequency min = {} Hz,  w_low = {}'.format(np.min(freq_var), w_low))
        print('frequency max = {} Hz , w_high = {}'.format(np.max(freq_var), w_high))
        print('T_min = {}, T_max = {}'.format(np.min(T),np.max(T)))
    
    remove_nan = False
    if remove_nan:
        T = T[~np.isnan(T)]
        T[T >= 1E308] = 0
    
    N = len(t)
    
    '''
    from waveleletAnalysis.py, COI
        # cone-of-influence, anything "below" is dubious
        plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",edgecolor="#00000040", hatch='x')
    '''
    
    if COI:
        # COI using function of time
        # y1 and y2 define coi line
        # b = t[N//2] # average time
        b = np.mean(t)
        C = 1/(np.max(t) - np.min(t))     #1E-6 # max freq of COI
        COI_func = COI_factor * C * (np.abs(b/t[t>0]) - 1) + np.min(freq_var)
        # COI_func = ((C/b) * np.abs((t/b - b)) + freq_var[-1])
        
        # COI using expression from Torrence + Compo 1998, COI = sqrt(2)*scale
        
        
        
        # COI using function of freq
        # COI_func = 1E-5 * 1.4142135623730951 * freq_var
        
    if CI: # for now this is limited to 95% confidence ie 5% significance
        chi_sqr_25 = 7.37775891 # chi sqr with DOF = 2 and p = 0.025, from https://www.danielsoper.com/statcalc/calculator.aspx?id=12
        chi_sqr_975 = 0.05063562 # chi sqr with DOF = 2 and p = 0.975
        
        confidence_interval_1 = (2/chi_sqr_25) * T
        confidence_interval_2 = (2/chi_sqr_975) * T
        # confidence_levels = np.array([confidence_interval_1,confidence_interval_2])
        # print('confidence interval: [{}, {}]'.format(confidence_interval_1,confidence_interval_2))
        # print('shapes: {}, {}'.format(np.shape(confidence_interval_1),np.shape(confidence_interval_2)))
        # fig_c, ax_c = plt.subplots()
        # cs_c = ax_c.contourf(t,freq_var,confidence_interval_2,levels = 20,cmap='Reds',extend = 'both', alpha = 1, locator=ticker.LogLocator(), vmin = np.min(confidence_interval_1),
        #vmax = np.max(confidence_interval_1))
        # ax_c.set_yscale('log')
        # ax_c.set_ylim([np.min(freq_var),np.max(freq_var)])
        # plt.show()
        
    
    # CONTOUR MAP
    if plot:
        # fig = plt.figure()
        Q = np.histogram(T,bins = 100)
        # vmin_ = Q[1][1] #np.percentile(T, q = 0.05) #np.min(T)
        vmin_ = np.percentile(T, q = vmin_q)
        vmax_ = np.max(T)
        cmap_ = 'viridis' #'cool' #'Reds'
        fig, ax = plt.subplots()
        cs = ax.contourf(t,freq_var,T,levels = levels_,cmap=cmap_,extend = 'both', alpha = 1, locator=ticker.LogLocator(), vmin = vmin_, vmax = vmax_) #vmin = np.min(T)
        # cs = ax.contourf(t,freq_var,T,levels,cmap='Reds',extend = 'both', alpha = 1)
        # cs = ax.imshow(T, aspect='auto', interpolation = 'gaussian',cmap = 'cool',extent=[np.min(t), np.max(t), np.min(f), np.max(f)]);

        if COI:
            ax.fill_between(t[t>0], y1 = 0, y2 = COI_func, facecolor="white",edgecolor="gray", hatch='/',alpha = 0.5)
            ax.fill_between(t[t>0], y1 = 0, y2 = np.flip(COI_func), facecolor="white",edgecolor="gray", hatch='/',alpha = 0.5)
            
            # ax.fill_betweenx(freq_var[freq_var > 0], x1 = 0, x2 = COI_func, facecolor="orange",edgecolor="gray", hatch='/',alpha = 0.5)
            # ax.fill_betweenx(freq_var[freq_var > 0], y1 = 0, y2 = np.flip(COI_func), facecolor="orange",edgecolor="gray", hatch='/',alpha = 0.5)
            
            # plt.fill_between(t, y1 = COI_func, y2 = 0, facecolor="none",edgecolor="blue", hatch='x')

        cbar = fig.colorbar(cs, label = r'$|T(t,f)|^2$', format = '%.3f')
        cbar.set_label(r'Wavelet Power: $|T(t,f)|^2$')
        
        if CI: # plot confidence interval contour lines
        
            ax.contour(t,freq_var,confidence_interval_1,levels = 1,colors = 'blue', linewidths = 0.5)
            ax.contour(t,freq_var,confidence_interval_2,levels = 1,colors = 'green', linewidths = 0.5)
            
        
        if LC_name:          
            ax.set_title('CWT Power Spectrum: {}, Wavelet basis: {}, {} freq bins'.format(LC_name,wavelet_type,len(freqs)))
        elif LC_name == False:
            ax.set_title('Simulated CWT Power Spectrum, Wavelet basis: {}, {} freq bins'.format(wavelet_type,len(freqs)))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_yscale('log')
        ax.set_ylim([np.min(freq_var),np.max(freq_var)])
        plt.show()
    
    '''
    use this code to plot with imshow instead of contourf (remember to flip the f vector from matlab):
        fig, ax = plt.subplots();
        cs = plt.imshow(T_plot, aspect='auto', extent=[np.min(t), np.max(t), f[1], np.max(f)]);
        ax.plot(t,coi);
        ax.set_yscale('log');
        plt.show()
    '''
    
    
    return(T,freq_var)

######################################################################################################################################

def calc_psd(x,t, plot = True, LC_name = False, normalize = 'rms squared', plot_low_freq_cuttof = 3, scale_by_freq = False):
    
    N = np.size(x) # this works if x is 1 LC
    # N = np.shape(x)[1] # this is if x is ensemble of K LC's
    sample_spacing = (np.max(t) - np.min(t))/N
    
    x_FT = np.fft.fft(x) # get FT
    ν = np.fft.fftfreq(n = N, d = sample_spacing) # get FT freqs
    
    if normalize == 'rms squared':
        P = ((2*sample_spacing) / (np.abs(np.mean(x))*N) ) * np.conj(x_FT)*x_FT # get power spectrum aka PSD
    
    # keep only positive frequencies
    if True:
        ν = ν[:N//2]
        P = P[:N//2]
    
    if scale_by_freq:
        P = ν * P
    
    P = np.real(P) # convert to real (imaginary part is 0 from ψ*ψ)
    
    # PLOT POWER SPECTRUM
    if plot:
        plt.figure()
        # plt.plot(ν[:N//2],P[:N//2])
        if plot_low_freq_cuttof:
            plt.plot(ν[ν > plot_low_freq_cuttof/np.max(t)],P[ν > plot_low_freq_cuttof/np.max(t)])
        else:
            plt.plot(ν,P)
        if LC_name:
            plt.title('Power spectrum: {}'.format(LC_name))
        else:
            plt.title('Power Spectrum')
        plt.xlabel('Frequency / Hz')
        plt.ylabel('Power')
        # plt.ylim(np.min(P[:N//2]),np.max(P[:N//2]))
        plt.xscale('log')
        plt.yscale('log')
        #plt.hlines(y = 0.426722,xmin = np.min(ν), xmax = np.max(ν)) # line for poisson noise for rev 3044
        # print('LC: {}: min = {}, max = {}'.format(LC_name,np.min(P[:N//2]),np.max(P[:N//2])))
        plt.show()
    
    return(P,ν)

######################################################################################################################################

def calc_pdf(x,t,num_bins = 30, LC_name = False, plot = False,normalize = True):
    
    # returns bin axis on the x, and normalized pdf on the y
    bin_axis = np.linspace(np.min(x),np.max(x),num_bins)
    pdf = np.histogram(x, bins = num_bins)[0]
    
    if normalize:
        pdf = pdf/np.sum(pdf) # normalize pdf
    
    if plot:
        plt.figure()
        plt.plot(bin_axis,pdf,'ko-')
        if LC_name:
            plt.title('PDF of {}, N = {}, {} bins'.format(LC_name,len(x),num_bins))
        else:
            plt.title('PDF of LC, N = {}, {} bins'.format(len(x),num_bins))
        plt.show()
    
    return(bin_axis,pdf)

######################################################################################################################################

def frequency2scale(f,t,wavelet_type = 'morl'):
    '''
    given the frequency, calc the scale to input to pywt.cwt()
    '''
    
    S_P = (np.max(t) - np.min(t))/len(t) # sampling period
    scale = pywt.central_frequency(wavelet = wavelet_type) / (f * S_P)
    
    return(scale)

######################################################################################################################################

def generate_poisson_noise(x,t, plot = False, noise_level = 1):
    
    #print(np.isnan(x))
    #x = x[~np.isnan(x)]
    #np.nan_to_num(x)
    N = np.size(x)
    #print('size = {}'.format(N))
    Δt = (np.max(t) - np.min(t))/N
    μ = Δt * np.abs(x)
    #μ = Δt * np.abs(x,dtype=np.float64) # mean for poisson dist is each point in x times spacing
    μ = μ / noise_level
    #μ.astype(np.float64) # this is to fix the error: ValueError: lam value too large
    #x_pois = (1/Δt) * np.random.poisson(lam = μ)
    x_pois = (1/Δt) * poisson.rvs(μ)
    
    return(x_pois)

######################################################################################################################################

def read_flc(f, plot = False, trunc_non_zero = True):
    '''
    given a .flc file f, output the time and counts, and plot
    '''

    # open the data set
    # FILENAME = lightcurves[0]
    FILENAME = f
    lcfits = fits.open(FILENAME)
    # print(lcfits)
    
    # get structure of file
    #lcfits.info()
    
    # get the data
    data = lcfits[1].data
    
    time = data.field('TIME')
    counts = data.field('RATE1')
    sig_err = data.field('ERROR1')
    
    counts = counts[np.logical_not(np.isnan(counts))] # remove NANs
    sig_err = sig_err[np.logical_not(np.isnan(sig_err))] # remove NANs
    
    N_counts = len(counts)
    time = time[:N_counts] # match size of time with counts, remove last
    lcfits.close()
    
    x = counts
    t = time
    
    # remove negative count rates
    if trunc_non_zero == True:
        x_nonzero = x[np.where(x > 0)]
        t = t[np.where(x > 0)]
        x = x_nonzero
        # print('trunc')
        # x[x < 0] = 0
    
    f = f.replace('/Users/akshayghosh/wavelet_analysis/lightcurves/','')
    f = f.replace('.flc','')
    
    # PLOT
    if plot:
        plt.figure()
        plt.title(f)
        # plt.plot(time,counts,'bo',markersize = '3')
        plt.errorbar(time,counts,yerr = sig_err)
        plt.xlabel('Time (s)')
        plt.ylabel('Count Rate')
        plt.show()
    
    LC_name = f
    
    return(x,t,LC_name)

######################################################################################################################################

def csv_to_lcdata(f):
    '''
    given a csv file that was created from a .fits file, return its data
    '''
    lc_data = pd.read_csv(f)
    
    t = lc_data['TIME'].to_numpy()
    
    x = lc_data['RATE'].to_numpy()
    x_err = lc_data['RATE.ERR'].to_numpy()
    
    bg = lc_data['BKG'].to_numpy()
    bg_err = lc_data['BKG.ERR'].to_numpy()
    
    # obs_title = lc_data['name'].to_numpy()[0]
    
    obs_title = 'name'#lc_data['name'].to_numpy()[0]
    
    # return(x,x_err,t)
    return(x,t,bg,x_err,bg_err,obs_title)
    
def csv_to_lcdata_suzaku(f):
    '''
    given a csv file that was created from a .fits file, return its data
    '''
    lc_data = pd.read_csv(f)
    
    t = lc_data['TIME'].to_numpy()
    
    x = lc_data['RATE'].to_numpy()
    x_err = lc_data['RATE.ERR'].to_numpy()
    
    # bg = lc_data['BKG'].to_numpy()
    # bg_err = lc_data['BKG.ERR'].to_numpy()
    
    # obs_title = lc_data['name'].to_numpy()[0]
    return(x,x_err,t)
    # return(x,t,bg,x_err,bg_err,obs_title)

######################################################################################################################################

def plot_lightcurve(x,t,LC_name = False, bin_info = True,trunc_non_zero = False, time_unit = 'ks'):
    
    bin_width = np.round((np.max(t) - np.min(t))/len(t),2)
    fig = plt.figure()
    if LC_name:
        if bin_info:
            plt.title(r'{}, $\Delta t = {}$ {}'.format(LC_name,bin_width, 's'))
        else:
            plt.title(LC_name)
    else:
        if bin_info:
            plt.title(r'Lightcurve, $\Delta t $ = {} {}'.format(bin_width, 's'))
        else:
            plt.title('Lightcurve')
    if time_unit == 'ks':
        plt.plot(t/1000,x)
    else:
        plt.plot(t,x)
    # plt.xlabel('Time (s)')
    plt.xlabel('Time / {}'.format(time_unit))
    plt.ylabel('Count Rate')
    plt.ylim([0,np.max(x)])
    # if trunc_non_zero:
    #     plt.ylim([0,np.max(x)])
    plt.show()
    
    return(fig)

def plot_psd(P,ν,LC_name = False):
    
    fig = plt.figure()
    if LC_name:
        plt.title(LC_name)
    else:
        plt.title('PSD')
    plt.plot(ν,P)
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Power')
    plt.xscale('log')
    plt.yscale('log') 
    plt.show()
    
    return(fig)

def plot_cwt(T,t,f,LC_name = False,levels = 20):
    
    # CONTOUR MAP
    fig = plt.figure()
    plt.imshow(T, aspect='auto', extent=[np.min(t), np.max(t), np.max(f), np.min(f)])
    # plt.contourf(t,f,T,levels,cmap='jet',extend = 'both') # extend to remove whitespace, can be 'both' or 'max
    plt.colorbar() #plt.colorbar(spacing = 'uniform')
    if LC_name:          
        plt.title('CWT Power Spectrum: {}, {} freq bins'.format(LC_name,len(f)))
    elif LC_name == False:
        plt.title('Simulated CWT Power Spectrum')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.show()
       
    return(fig)

def plot_pdf(x,t,num_bins = 20, LC_name = False):
    
    bin_axis = np.linspace(np.min(x),np.max(x),num_bins)
    pdf = np.histogram(x, bins = num_bins)[0]
    pdf = pdf/np.sum(pdf) # normalize pdf
    
    plt.figure()
    plt.plot(bin_axis,pdf,'ko-')
    if LC_name:
        plt.title('PDF of {}, N = {}, {} bins'.format(LC_name,len(x),num_bins))
    else:
        plt.title('PDF of LC, N = {}, {} bins'.format(len(x),num_bins))
    plt.show()
    
    return()

######################################################################################################################################

def adjust_bin_width(x,t,new_bin_width):
    
    n = len(x)
    initial_bin_width = (np.max(t) - np.min(t))/n
    q = new_bin_width/initial_bin_width # q is bin_multiplier
    
    x2 = np.reshape(x,(int(n/q),int(q)))
    t2 = np.reshape(t,(int(n/q),int(q)))
    
    x2_avg = np.mean(x2,axis = 1)
    t2_avg = np.mean(t2,axis = 1)
    
    x_adjust = np.reshape(x2_avg,len(x2_avg))
    t_adjust = np.reshape(t2_avg,len(t2_avg))
    
    # x_adjust = np.concatenate(x2_avg)
    # t_adjust = np.concatenate(t2_avg)

    return(x_adjust,t_adjust)

######################################################################################################################################

def calc_global_cwt(T,t,freq, plot = True, normalize = True):
    # given a cwt T, func of t,f, calc the integral over t to get the global wavelet transform

    method = 'sum'
    N = len(t)
    
    if method == 'integral':
        G = np.trapz(y = T, x = t, axis = -1)
        if normalize:
            G = G / np.sum(G)
    
    if method == 'sum':
        G = (1/N) * np.sum(T,axis = 1)

    if plot:
        plt.figure()
        plt.plot(freq,G)
        plt.title('Global Wavelet Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavelet Power')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    return(G)

######################################################################################################################################

def add_periodic_gaps(x,t,num_gaps,gap_length, random_gap_length = 0, plot = True, random_gaps = False, gap_type = 'zero'):
    
    # x[int(0.2*N):int(0.5*N)] = -1
    
    N = len(t)
    q = np.max(t) - np.min(t)
    k = num_gaps
    L = gap_length/k
    y_max = np.max(x)
    x_max = np.max(x)
    x_g = x.copy()
    t_g = t.copy()
    
    for i in range(k):
        n = i + 1
        gap_start = n/(k + 1) - L/2
        gap_end = n/(k + 1) + L/2
        # x[int(gap_start*N):int(gap_end*N)] += -x_max - 10
        if gap_type == 'nan':
            x_g[int(gap_start*N):int(gap_end*N)] = np.NaN
        if gap_type == 'zero':
            x_g[int(gap_start*N):int(gap_end*N)] = 0
        print('gap {} s to {} s'.format(gap_start*q,gap_end*q))
        
    if random_gaps:
        random_gap_start = np.random.uniform(low = 0, high = 1 - random_gap_length)
        random_gap_end = random_gap_start + random_gap_length
        # x[int(random_gap_start*N):int(random_gap_end*N)] += -x_max - 10
        x_g[int(random_gap_start*N):int(random_gap_end*N)] = np.NaN
        
    if plot:
        plt.figure()
        plt.scatter(t_g,x_g,s = 4)
        # plt.plot(t,x)
        plt.xlabel('Time (s)')
        plt.ylabel('Count rate')
        # plt.title('Lightcurve with {} gaps'.format(k))
        plt.title('Lightcurve with gaps')
        plt.ylim([0,y_max])

    return(x_g,t_g)

######################################################################################################################################

def python_to_matlab(x,t,folder_path = '/Users/akshayghosh/wavelet_analysis/matlab_files/', LC_name = False, timestamp = False):
    
    # # need data to be column vectors
    # x = x[:,None]
    # t = t[:,None]
    
    if LC_name == False:
        data_dictionary = {'count_rate':x,'time':t}
        if timestamp == True:
            filename_out = '{}lc_{}.mat'.format(folder_path,datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
        elif timestamp == False:
            filename_out = '{}lc.mat'.format(folder_path)
    else:
        data_dictionary = {'name': LC_name, 'count_rate': x, 'time': t}
        if timestamp == True:
            filename_out = '{}{}_{}.mat'.format(folder_path,LC_name,datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
        elif timestamp == False:
            filename_out = '{}{}.mat'.format(folder_path,LC_name)
    
    # if timestamp == True:
    #     filename_out = '{}/lightcurve_{}.mat'.format(folder_path,datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    # elif timestamp == False:
    #     filename_out = '{}/lightcurve_{}.mat'.format(folder_path,LC_name)
    
    savemat(filename_out,mdict = data_dictionary)
    
    return()

######################################################################################################################################

def python_to_matlab_fits(t,t_err,x,x_err,bg,bg_err,name,folder_path):
    
    '''
    MATLAB code to read this:
        
        % get info from data structure
        t = Q.TIME;
        t_err = Q.TIME_ERR;
        x = Q.RATE;
        x_err = Q.RATE_ERR;
        bg = Q.BKG;
        bg_err = Q.BKG_ERR;
        %obs_title = char(Q.name(1)); % cellstr(Q.name), obs_title = cellstr(Q.name)
        obs_title_vec = cellstr(Q.name);
    '''
    
    # nested dict to match structure of mat files from fits files from R
    data_dictionary = {'lc_data' : {'TIME': t.T, 'TIME_ERR': t_err.T, 'RATE': x.T, 'RATE_ERR': x_err.T, 'BKG': bg.T, 'BKG_ERR': bg_err.T, 'name': name}}
    filename_out = '{}{}.mat'.format(folder_path,name)
    
    savemat(filename_out,mdict = data_dictionary)
    
    return()

######################################################################################################################################

def matlab_to_python(m_file):
    
    # take a matlab file and return its wt, coi, f, t
    
    Y = loadmat('/Users/akshayghosh/wavelet_analysis/matlab_files/1Zw1_matlab/cwt_LC_3680.mat')

    return()

######################################################################################################################################

def impute_gaps(x,t, plot = True,method = 'simple'):
    
    '''
    given an array x with missing values represented by 'NaN', imppute the
    missing values and return the imputed array
    '''
    
    if method == 'simple':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_impute = imp.fit(x.reshape(-1, 1))
    
    if method == 'KNN':
        imp = KNNImputer(n_neighbors = 3, weights = 'distance')
        x_impute = imp.fit(x.reshape(-1, 1))       
    
    
    x_imputed = imp.transform(x.reshape(-1, 1))
    x_return = x_imputed.reshape(int(len(x)),) # reshape back to original shape
    
    if plot:
        Q = plot_lightcurve(x_imputed,t, LC_name = 'Imputed Lightcurve')  
    
    return(x_return)



######################################################################################################################################

def generate_simulated_lc_ensemble(FILENAME_in,LC_NAME_out, b = 2.0, K = 1000,file_type = 'csv',
                                   output_folder = '/Users/akshayghosh/wavelet_analysis/matlab_files/IRAS13224_3809_rev3044_sim_ensemble',
                                   qpo = False,custom_N = False,custom_t_stop = False,custom_scale = False, non_uniform_sampling = False,FILENAME_bg_nustar = None):
    '''
    generate an ensemble of K lightcurves resembling an input lightcurve with a set power law
    '''
    
    #FILENAME = '/Users/akshayghosh/wavelet_analysis/matlab_files/IRAS13224_lc/IRAS_132243809_2127_0673580201_lc.flc'
    FILENAME = FILENAME_in
    
    if file_type == 'flc':
        x_obs,t_obs,name = read_flc(FILENAME,plot = False, trunc_non_zero = True)
    
    if file_type == 'csv':
        x_obs,t_obs,bg,x_err,bg_err,obs_title = csv_to_lcdata(FILENAME)
        # x_obs,x_err,t_obs = csv_to_lcdata_suzaku(FILENAME) # changed this line for the suzaku LCs
        
    if file_type == 'dat':
        # open .dat file
        if False:
            # this is specific to NGC 6814, XMM 2016
            with open(FILENAME_in, 'r') as file:
                data = np.loadtxt(file,usecols = (0,1,2),skiprows = 1)
            # t_obs = 1000 * data[:,0]
            # x_obs = data[:,1]
            #x_err = data[:,2]
            t_obs = data[:,0]
            x_obs = data[:,1]
            x_err = data[:,2]
            # t_cut = 1.2446e5
            # x_obs = x_obs[t_obs < t_cut]
            # x_err = x_err[t_obs < t_cut]
            # t_obs = t_obs[t_obs < t_cut]
        if False:
            # this is specific to NGC 6814, Swift 2022
            with open(FILENAME_in, 'r') as file:
                data = np.loadtxt(file,usecols = (0,1,2),skiprows = 1)
            t_obs = 86400 * data[:,0]
            x_obs = data[:,1]
            # x_err = data[:,2]
        if True:
            # specific for MRK 335, swift 2022
            # construct t,x,x_err vectors, removing values where x < 0
            with open(FILENAME_in, 'r') as file:
                data = np.loadtxt(file,usecols = (0,1,2))
            t_obs = 86400 * np.delete(data[:,0], np.where(data[:,1] < 0))
            x_obs = np.delete(data[:,1], np.where(data[:,1] < 0))
            x_err = np.delete(data[:,2], np.where(data[:,1] < 0))
    
    if file_type == 'fits':
        t_obs,x_obs,x_err = read_fits_suzaku(FILENAME)
        
    if file_type == 'fits_nustar':
        t_obs,x_obs,x_err,bg,bg_err = read_fits_nustar(FILENAME, FILENAME_bg_nustar)
    
    # t_cut = 60000
    # x_obs = x_obs[t_obs < t_cut]
    # x_err = x_err[t_obs < t_cut]
    # t_obs = t_obs[t_obs < t_cut]
    
    # create pdf of counts for smiluation
    counts_obs,pdf_obs = calc_pdf(x_obs,t_obs,num_bins = 30, LC_name = False, plot = False)
    
    
    # create array of shape [num_lightcurves,data_points] to save each realization of the sim
    if type(custom_N) == bool:
        data_points_ = len(x_obs)
    else:
        data_points_ = custom_N
        
    # data_points_ = 2000
    num_lightcurves = K
    
    if type(custom_scale) == bool:
        # scale_ = np.max(x_obs)
        scale_ = np.mean(x_obs)
    else:
        scale_ = custom_scale
    # print('time = {}'.format(np.max(t_obs)))
    
    if type(custom_t_stop) == bool:
        t_stop_ = np.max(t_obs)
    else:
        t_stop_ = custom_t_stop
    
    # t_stop_ = np.max(t_obs)
    # t_stop_ = 200000
    x_sim_ensemble = np.zeros(shape = [num_lightcurves,data_points_])
    
    print('USING LC: {}'.format(FILENAME))
    print('{} data points per LC'.format(data_points_))
    print('LC DURATION: {} ks'.format(t_stop_ / 1000))
    print('OUTPUT FILE: {}{}.mat'.format(output_folder,LC_NAME_out))
    #num_lightcurves = 1000 # to simulate 1000 lightcurves
    
    # simulate num_lightcurves realizations of a LC resembling the observed LC
    for i in tqdm(range(num_lightcurves)):
        # print(i)
        #print('{} % complete...'.format(100*i/num_lightcurves))
        # for custom pdf, send array thats like input_pdf = (counts,probabilities)
        x_sim_ensemble[i],t_sim = E2013_LC(sections = 1, data_points = data_points_,plot = False, power_law = b, input_pdf = [counts_obs,pdf_obs], poisson_noise = True, num_iterations = 80,
                 scale = scale_, noise = 1, t_stop = t_stop_ , non_uniform_sampling = non_uniform_sampling)
        
        # ADD QPO TO LC
        # amp1 = 0.5;amp2 = 0.5;phase1 = 1e-3;phase2 = 1e-4;count_shift = 0
        # # # amp1 = 6;amp2 = 6;phase1 = 1e-3;phase2 = 1e-4;count_shift = 0
        # # # amp1 = 6/15;amp2 = 6/15;phase1 = 1e-3;phase2 = 1e-4;count_shift = 20
        # # amp1 = 1;amp2 = 1;phase1 = np.random.normal(loc = 1e-3, scale = 5e-5, size = len(t_sim));phase2 = np.random.normal(loc = 1e-4, scale = 5e-6, size = len(t_sim));count_shift = 0
        # qpo_component = 0.4*amp1*np.sin(2*phase1*pi*t_sim) + 0.3*amp2*np.sin(2*phase2*pi*t_sim) + count_shift
        # qpo_noise = np.random.normal(loc = np.mean(x_sim_ensemble[i]),scale = 0.5*np.var(x_sim_ensemble[i]),size = len(x_sim_ensemble[i]))
        # x_sim_ensemble[i] += qpo_component + qpo_noise # add qpo and qpo noise
        # x_sim_ensemble[i] = x_sim_ensemble[i] - np.min(x_sim_ensemble[i]) # shift so that all values are  +ve
    
        # if i == 0:
        #     # Q1 = plot_lightcurve(x_sim_ensemble[i],t_sim)
        #     Q2 = calc_psd(x_sim_ensemble[i],t_sim)
        
    #folder_path_ = '/Users/akshayghosh/wavelet_analysis/matlab_files/iras_simulated_LC_ensemble'
    #folder_path_ = '/Users/akshayghosh/wavelet_analysis/matlab_files/iras_simulated_LC_ensemble_accpl'
    #folder_path_ = '/Users/akshayghosh/wavelet_analysis/matlab_files/IRAS13224_3809_rev3044_sim_ensemble'
    folder_path_ = output_folder
    #LC_name_ = 'IRAS132243809_2127_sim_ensemble'
    LC_name_ = LC_NAME_out
    python_to_matlab(x_sim_ensemble,t_sim,folder_path = folder_path_, LC_name = LC_name_, timestamp = False)
    print('COMPLETE !!!!')


    return()

######################################################################################################################################

def sim_to_matlab_struct(x,t,folder_path = '/Users/akshayghosh/wavelet_analysis/matlab_files/', LC_name = False, timestamp = False):
    
    '''
    MATLAB CODE TO READ STRUCT:
    
    % get info from data structure
    t = Q.TIME;
    t_err = Q.TIME_ERR;
    x = Q.RATE;
    x_err = Q.RATE_ERR;
    bg = Q.BKG;
    bg_err = Q.BKG_ERR;
    obs_title = char(Q.name(1));
    '''
    '''
    For errors you might be able to just use like +/-2% or so of the mean count rate, maybe calculate the mean error size from the real 
    light curves you have and just use that.
    For background, you could actually just simulate white noise between 0 and 0.3 cts/sec and use that.
    '''
    
    '''
    SET UP LOGNORMAL DIST:
    lognormal_mean = np.log(np.square(μ_x) / np.sqrt(np.square(μ_x) + np.square(var_x)))
    lognormal_sigma = np.log(1 + np.square(var_x) / np.square(μ_x))
    x_sim_1 = np.random.lognormal(mean = lognormal_mean, sigma = lognormal_sigma, size = N) # lognormal distribution
    '''
    
    # if timestamp == True:
    #     filename_out = '{}lc_{}.mat'.format(folder_path,datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    
    # need data to be column vectors
    x = x[:,None]
    t = t[:,None]
    
    # set errors, background
    n = len(x)
    t_err = ((np.max(t) - np.min(t))/n) * np.ones(shape = (n,1))#np.zeros(shape = (n,1))
    
    x_err_ratio = 4.5/100
    x_err = np.random.normal(loc = x_err_ratio * np.mean(x), scale = 0.025, size = (n,1))#np.ones(shape = (n,1))#np.zeros(shape = (n,1))
    
    # adjust x so that none of the errorbars go below zero
    min_vals = x - x_err # the lowest point of each error bars
    
    # print(len(np.argwhere(min_vals)))
    # print('earlier max x = {}'.format(np.max(x)))
    
    # if len(np.argwhere(min_vals)) > 0: # temp commented out cause it was messing up the LC amps
    #     x = x - np.min(min_vals)
    # # x = x - (np.min(x) - np.min(x_err))
    
    
    ##to simulate background, create white noise LC where the count rates are in [0,0.3] counts/s
    bg_max = 0.3/10
    bg_err_ratio = 0.35*10 # percentage of mean bg to use for bg errors
    # bg,t_bg = E2013_LC(sections = 1, data_points = n,plot = False, power_law = 0.0, input_pdf = 'lognormal', poisson_noise = True, num_iterations = 80,
    #           scale = False, noise = 1, t_stop = np.max(t),qpo = False)#np.zeros(shape = (n,1))
    
    fn_bg = '/Users/akshayghosh/wavelet_analysis/matlab_files/obs_files_other_sources/ark120_0147190101_lccor_300-10000.csv'
    x_obs,t_obs,bg_obs,x_err_obs,bg_err_obs,obs_title = csv_to_lcdata(fn_bg)
    counts_obs,pdf_obs = calc_pdf(bg_obs,t_obs,num_bins = 30, LC_name = False, plot = False) # input_pdf = [counts_obs,pdf_obs]
    
    bg,t_bg = E2013_LC(sections = 1, data_points = n,plot = False, power_law = 0.0, input_pdf = [counts_obs,pdf_obs], poisson_noise = True, num_iterations = 80,
              scale = 1.1, noise = 1, t_stop = 140000)
    
    
    bg = match_limits(vec_to_scale = bg,vec_to_match = [0,bg_max])
    
    bg_err = bg_err_ratio * np.mean(bg) * np.ones(shape = (n,1))#np.zeros(shape = (n,1))
    bg = bg - (np.min(bg) - bg_err[0])

    obs_title = [LC_name]*n
    
    data_dictionary = {'TIME':t,'TIME_ERR':t_err,'RATE':x,'RATE_ERR':x_err,'BKG':bg,'BKG_ERR':bg_err,'name':obs_title}
    lc_data_dict = {'lc_data':data_dictionary} # need dictionary in dictionary because of matlab code
    
    # print('later max x = {}'.format(np.max(x)))
    
    if timestamp == True:
        filename_out = '{}/{}_{}.mat'.format(folder_path,LC_name,datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    else:
        filename_out = '{}/{}.mat'.format(folder_path,LC_name)
        
    savemat(filename_out,mdict = lc_data_dict)
    
    return()

######################################################################################################################################

def generate_qpo(N,t_stop,b,cauchy_centers,cauchy_amps,cauchy_scales, plot = False, qpo = True, normalize = False):
    '''
    the parameters for amplitude and scale are actually 10^amplitude and 10^scale
    '''
    
    # b = 2.0
    # t_start = 0;t_stop = 140000
    # N = 1300
    t_start = 0
    
    # print(type(cauchy_centers))
    # num_lorentzians = len(cauchy_centers)
    
    w_start = 1 / (t_stop - t_start)
    w_stop = N / (t_stop - t_start)
    w = np.linspace(w_start,w_stop,N)
    
    S = np.power(w,-b) # add lorentzians to this
    
    if normalize:
        S = S / auc(w,S)
    
    cauchy_signal = np.zeros(N)

    if qpo:
        if type(cauchy_centers) == np.ndarray or type(cauchy_centers) == list:
            num_lorentzians = len(cauchy_centers)
            for sig in range(num_lorentzians):
                cauchy_center = cauchy_centers[sig]
                cauchy_amp = cauchy_amps[sig]
                cauchy_scale = cauchy_scales[sig]
                cauchy_signal += np.power(10.0,cauchy_amp) * cauchy.pdf(w, loc = cauchy_center,scale = np.power(10.0,cauchy_scale))
        
        else:
            cauchy_signal += np.power(10.0,cauchy_amps) * cauchy.pdf(w, loc = cauchy_centers,scale = np.power(10.0,cauchy_scales))
    
    # S = S / auc(w,S)
    # signal_ratio = auc(w,cauchy_signal) / auc(w,S)
    # print('LOR / PL power = {}, pl = {}'.format(signal_ratio,b))
    
    if plot:
        plt.figure()
        plt.plot(w,S + cauchy_signal)
        # plt.plot(w,S,'k')
        plt.plot(w,cauchy_signal,'r')
        plt.title('INPUT PSD');plt.xlabel('Freq');plt.ylabel('Amp');plt.xscale('log');plt.yscale('log')
        plt.show()
        
    S = S + cauchy_signal
    
    return(S)

######################################################################################################################################

def match_limits(vec_to_scale,vec_to_match):
    '''
    ADAPTED FROM THIS MATLAB CODE:
    
    function [A_scaled] = match_limits(vec_to_scale,vec_to_match)
    % scale A so that the min and max are the same as B
    
    A = vec_to_scale;
    B = vec_to_match;
    
    a = min(A);b = max(A);c = min(B);d = max(B);
    A_scaled = ((d-c)/(b-a))*A - (((d-c)/(b-a))*a - c);
    
    end
    '''
    # scale A (aka vec_to_scale) so that the min and max are the same as B (aka vec_to_match)
    A = vec_to_scale;
    B = vec_to_match;
    
    a = np.min(A);b = np.max(A);c = np.min(B);d = np.max(B);
    A_scaled = ((d-c)/(b-a))*A - (((d-c)/(b-a))*a - c);
    
    
    return(A_scaled)

######################################################################################################################################

def calc_wwz(t,x,x_err,LC_name = False, plot = True,
             decay_const_multiplier = 1, freq_method_ = 'log', WWZ_METHOD = 'Kirchner_f2py',
             num_levels_ = 100, v_percentiles = [0.1,97.5], 
             calc_LSP = True,tau=None, num_freqs = 100,freq_range = 'auto',
             plane_of_sig = False, fn_sig = False ,sig_levels = False,print_M_progress = False,
             custom_num_lc = False,save_plot = False, LSP_use_x_err = True,LC_errorbar = True):

    default_decay_const = 1/(8*np.pi**2) # freq_method='lomb_scargle'
    
    # if tau == 'auto':
    #     tau = np.linspace(np.min(t),np.max(t),len(t))
    # else:
    #     tau = None
    
    # WWZ_METHOD = 'Kirchner_f2py' # 'Kirchner_numba' for non fortran; 'Kirchner_f2py' for fortran
    
    # freq_bass = 4.749578474910352e-06#1/(np.max(t) - np.min(t)) # 1.3e-5 for ngc6841 2016 xmm data
    # freq_treble = 1e-2#1/np.min(np.histogram(np.diff(t),bins = 50)[1])
    # print('freq range = ',freq_bass,freq_treble)
    
    if freq_range == 'auto':
        freq_bass = 1/(np.max(t) - np.min(t)) # 1.3e-5 for ngc6841 2016 xmm data
        freq_treble = 1/np.min(np.histogram(np.diff(t),bins = 50)[1])
    else:
        freq_bass = freq_range[0]
        freq_treble = freq_range[1]
    
    # freq_ = None
    # freq_ = np.linspace(freq_bass,freq_treble,1000)
    
    # num_freqs = 100 # this was default at 100
    freq_ = np.geomspace(freq_bass,freq_treble,num_freqs)
    
    # tau = np.linspace(np.min(t),np.max(t),len(t)) # setting tau like this makes the code way too slow

    res = pyleo.utils.wavelet.wwz(ys = x, ts = t, tau = tau, ntau=None, freq=freq_, freq_method = freq_method_, 
                            freq_kwargs={}, c = decay_const_multiplier*default_decay_const, Neff_threshold=3, 
                            Neff_coi=3, nproc=8, detrend=False, sg_kwargs=None, method=WWZ_METHOD, 
                            gaussianize=False, standardize=False, len_bd=0, bc_mode='reflect', reflect_type='odd')
    
    W = res.amplitude.T # wwz coeffs are real, not complex like the cwt?
    W = np.square(W)
    W_time = res.time
    W_freq = res.freq
    W_per = 1/res.freq
    COI_freq = 1/res.coi
    
    # get max non-nan freq of WWZ transform
    # np.isnan: True where x is NaN, false otherwise. This is a scalar if x is a scalar.
    nan_idx_vec = np.argmax(np.isnan(W), axis=0) # each element in this vector is the idx of the first NaN value of each col of W
    # wwz_freq_max_idx = int(np.mean(nan_idx_vec) + np.std(nan_idx_vec,ddof = 1))
    wwz_freq_max = np.max(W_freq)#W_freq[wwz_freq_max_idx]
    
    # calc vmin and vmax based off the 0.1th and 97.5th percentile of W
    W_flat = W.flatten() # flatten W to remove nans
    W_flat = W_flat[~np.isnan(W_flat)] # remove nans to calc percentiles
    vmin_,vmax_ = np.percentile(W_flat, q = (v_percentiles[0],v_percentiles[1]))
    
    
    # PSD,f = av.calc_psd(x, t, plot = False)
    
    if calc_LSP:
        if all(np.diff(t)==np.diff(t)[0]) == False: # if not evenly spaced calc LSP
            LSP_min_freq = np.min(W_freq)
            LSP_max_freq = np.max(W_freq)
            if LSP_use_x_err:
                f, PSD = LombScargle(t, y = x, dy = x_err).autopower(minimum_frequency=LSP_min_freq,maximum_frequency=LSP_max_freq) # LSP from astropy
            if LSP_use_x_err == False:
                f, PSD = LombScargle(t, y = x, dy = None).autopower(minimum_frequency=LSP_min_freq,maximum_frequency=LSP_max_freq)
            psd_type = 'LSP' # for the axis label
        if all(np.diff(t)==np.diff(t)[0]) == True:
            PSD,f = calc_psd(x,t, plot = False, LC_name = False, normalize = 'rms squared', plot_low_freq_cuttof = 3, scale_by_freq = False)
            PSD = np.real(PSD)
            psd_type = 'PSD' # for the axis label
    if calc_LSP == False:
        psd_type = 'None'
    
    
    ''
    # Visualization
    if plot:
        # plt.style.use('classic')
        # plt.style.use('ggplot')
        plt.style.use('seaborn-v0_8-ticks')
        plot_size = 10.5 # <----- ********USE THIS TO ADJUST SIZE OF PLOTS!!!!!!!!!!********
        font_size_axes = 25 # <----- *****USE THIS TO ADJUST SIZE OF AXIS FONT!!!!!!!!!!****
        font_size_titles = 17 # <----- ***USE THIS TO ADJUST SIZE OF TITLE FONT!!!!!!!!!!***
        axis_font_size = 20
        axis_tick_size = 20
        
        num_levels = 100
        levels_ = np.linspace(vmin_,vmax_,num_levels)

        fig = plt.figure(figsize=(12,12))
        
        gs = gridspec.GridSpec(nrows=2, ncols=2,wspace=0,hspace=0, width_ratios=[1, 3], height_ratios=[3, 1],figure = fig)
        ax_wwz = plt.subplot(gs[0, 1])
        ax_LC = plt.subplot(gs[1,1])
        ax_PSD = plt.subplot(gs[0,0])
        # ax_wwz = plt.subplot(gs[0, 1], sharey = ax_PSD)
        
        if plane_of_sig:
            M = calc_plane_of_sig(fn_sig = fn_sig ,sig_levels = sig_levels,
                                  decay_const_multiplier = decay_const_multiplier, WWZ_METHOD = WWZ_METHOD,
                                  print_M_progress = print_M_progress,tau = tau,custom_num_lc = custom_num_lc, freq_method=freq_method_,freq=freq_)
            M_contour = np.zeros(np.shape(M))
            
            # print(f'M_contour SHAPE = {np.shape(M_contour)}')
            # print(f'M SHAPE = {np.shape(M)}')
            # print(f'W SHAPE = {np.shape(W)}')
            

            for idx,sig in enumerate(sig_levels):
                linestyle_vec = ['dashed','solid']
                M_contour[idx,:,:][W > M[idx,:,:]] = 1
                cont_sig = ax_wwz.contour(W_time, W_freq, M_contour[idx,:,:], levels = 0, colors = 'black', linewidths = 0.8, linestyles = linestyle_vec[idx])
        
        contourf_args = {'cmap': 'Reds',
                          'origin': 'lower',
                          'levels': levels_,
                          'extend': 'both'}
        cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
        cont = ax_wwz.contourf(W_time, W_freq, W, **contourf_args)
        
        ax_wwz.plot(W_time, COI_freq, 'k--')  # plot the cone of influence
        
        # shade under COI
        ax_wwz.fill_between(W_time, y1 = 0, y2 = COI_freq, facecolor="white",edgecolor="gray", hatch='/',alpha = 0.3)
        ax_wwz.fill_between(W_time, y1 = 0, y2 = np.flip(COI_freq), facecolor="white",edgecolor="gray", hatch='/',alpha = 0.3)
        
        
        ax_wwz.set_yscale('log')
        # ax_wwz.set_ylim([np.min(W_freq),wwz_freq_max])
        ax_wwz.set_xlabel('Time / s')
        ax_wwz.set_ylabel('Frequency / Hz')
        # ax_wwz.set_title('WWZ Transform: MRK 335 Swift LC')
        # cb = plt.colorbar(cont, **cbar_args)
        # @savefig wwa_wwz.png
        ax_wwz.set_facecolor('black')
        ax_wwz.grid(False)
        ax_LC.grid(False)
        ax_PSD.grid(False)
        
        if LC_errorbar == True:
            ax_LC.errorbar(t,x,yerr = x_err,capsize = 0,fmt = 'k.',markersize = 2)
        elif LC_errorbar == False:
            ax_LC.plot(t,x,'ko',markersize = 4)
        '''
        plot fitted sine wave for NGC obs
        '''
        # if True:
        #     from scipy import optimize

        #     def test_func(x, a, b,c):
        #         return a * np.sin(2*pi*b * x) + c
            
        #     x_data = t[t < 60000]#t[t > 20000]
        #     y_data = x[t < 60000]#x[t > 20000]
        #     params, params_covariance = optimize.curve_fit(test_func, x_data, y_data,
        #                                                    p0=[2, 5e-5, 0.9])
        #     ax_LC.plot(x_data, test_func(x_data, params[0], params[1], params[2]),
        #              label='Fitted function',c = 'red',lw = 5)
        #     print(f'FITTED FREQ = {params[1]} Hz')
        
        if calc_LSP:
            ax_PSD.plot(PSD,f,'k-',lw = 0.6, alpha = 0.8)
            # ax_PSD.loglog(PSD,f,'k-',lw = 0.6, alpha = 0.8)
        ax_PSD.set_xscale('log');ax_PSD.set_yscale('log')
        ax_PSD.invert_xaxis()
        ax_PSD.xaxis.set_minor_locator(NullLocator())
        # ax_PSD.tick_params(axis='y', which='minor')
        # ax.yaxis.set_major_locator(plt.NullLocator())
        
        # wwz_ylim_min = np.min( [np.min(f),np.min(W_freq) ])
        # wwz_ylim_max = np.max( [np.max(f),np.max(W_freq) ])
        
        # wwz_ylim_min = np.min(W_freq)
        # wwz_ylim_max = np.max(W_freq)
        wwz_ylim_min = freq_bass
        wwz_ylim_max = freq_treble
        
        ax_wwz.set_ylim([wwz_ylim_min,wwz_ylim_max])
        ax_PSD.set_ylim([wwz_ylim_min,wwz_ylim_max])
        
        # ax_wwz.set_ylim([wwz_ylim_min,2.5e-5])
        # ax_PSD.set_ylim([wwz_ylim_min,2.5e-5])
        # ax_wwz.set_ylim([wwz_ylim_min,2e-6])
        # ax_PSD.set_ylim([wwz_ylim_min,2e-6])
        # ax_wwz.set_ylim([wwz_ylim_min,2e-3])
        # ax_PSD.set_ylim([wwz_ylim_min,2e-3])
        
        # ax_wwz.set_ylim([np.min(W_freq),wwz_freq_max])
        # ax_PSD.set_ylim([np.min(W_freq),wwz_freq_max])
        
        # # Hide X and Y axes label marks
        # ax_wwz.xaxis.set_tick_params(labelbottom=False)
        # ax_wwz.yaxis.set_tick_params(labelleft=False)
        
        # Hide X and Y axes tick marks
        # ax_wwz.set_xticks([])
        # ax_wwz.set_yticks([])
        ax_wwz.tick_params(axis = 'both', which = 'both', length = 0)
        

        # y_minor = plt.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        # ax_PSD.yaxis.set_minor_locator(y_minor)
        # ax_PSD.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        # from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        # # ax_PSD.yaxis.grid(True, which='minor')
        # ax_PSD.yaxis.set_minor_locator(AutoMinorLocator())
        
        # ax_PSD.minorticks_on()
        
        # ax_PSD.set_yticks(ticks = [1, 2 ,3], minor = False)
        # ax_PSD.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax_PSD.tick_params(axis='y', which='minor')
        # ax_PSD.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10)))
        
        # set axis labels
        ax_LC.set_xlabel('Time / s',fontsize = 1.2*axis_font_size)
        ax_LC.set_ylabel('Counts / s',fontsize = 1.2*axis_font_size)
        # ax_PSD.set_xlabel(f'Power: {psd_type}')
        ax_PSD.set_xlabel(f'{psd_type}',fontsize = 1.2*axis_font_size)
        ax_PSD.set_ylabel('Frequency / Hz',fontsize = 1.2*axis_font_size)
        
        
        ax_PSD.tick_params(axis='both', which='major', labelsize=axis_tick_size)
        # ax_PSD.tick_params(axis='both', which='minor', labelsize=axis_tick_size)
        # ax_PSD.set_xticklabels([100*np.min(PSD),0.01*np.min(PSD)])
        # try:
        #     psd_power_tick_vals = find_powers_of_10(np.min(PSD), np.max(PSD))
        # except ValueError:
        #     psd_power_tick_vals = find_powers_of_10(np.min(PSD) + 0.000001, np.max(PSD))
        
        if calc_LSP == True:
            psd_power_tick_vals = find_powers_of_10(np.min(PSD[PSD > 0]), np.max(PSD))
            ax_PSD.xaxis.set_ticks([psd_power_tick_vals[1],psd_power_tick_vals[-2]])
        
        ax_PSD.tick_params(direction='in', which = 'major',length=14, width=2, colors='k')
        ax_PSD.tick_params(direction='in', which = 'minor',length=8, width=2, colors='k')
        
        ax_LC.tick_params(axis='both', which='major', labelsize=axis_tick_size)
        ax_LC.tick_params(direction='in', which = 'major',length=14, width=2, colors='k')
        
        # print(f'psd range = {np.min(PSD)},{np.max(PSD)}')
        # from matplotlib.ticker import LogLocator
        # nticks = 9
        # maj_loc = ticker.LogLocator(numticks=nticks)
        # min_loc = ticker.LogLocator(subs='all', numticks=nticks)
        # ax_PSD.yaxis.set_major_locator(maj_loc)
        # ax_PSD.yaxis.set_minor_locator(min_loc)
        
        # # draw black lines around each plot
        # border_w = 4
        # ax_wwz.axvline(x=np.max(W_time), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        # ax_wwz.axvline(x=np.min(W_time), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        # # ax_wwz.axhline(x=np.max(W_time), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        # # ax_wwz.axhline(x=np.min(W_time), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        
        # ax_PSD.axvline(x=np.max(PSD), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        # ax_PSD.axvline(x=np.min(PSD), ymin=0.0, ymax=1.0, color='k', linestyle='-', alpha=1,lw = border_w)
        
        
        
        # plt.subplots_adjust(wspace=0, hspace=0) # remove gaps
        if LC_name:
            ax_wwz.set_title(f'WWZ Transform: {LC_name}')
            # plt.title(f'WWZ Transform: {LC_name}',loc='right')
        plt.tight_layout()
        plt.show()
        
        if save_plot:
            plt.savefig(save_plot)

    return(W_time,W_freq,W)

######################################################################################################################################

def plot_sampling_dist(t, bins = 50, LC_name = False):
    
    diff = np.diff(t)
    
    plt.figure()
    plt.hist(diff, bins = bins)
    plt.xlabel(r'$\Delta t$ / s')
    plt.ylabel('Counts')
    if LC_name:
        plt.title(f'Sampling distribution: {LC_name}')
    plt.show()

    return()


######################################################################################################################################

class TimeSeries:
    """
    Time Series class
    """
    def __init__(self, time, value, error, name='time_series'):
        self.time = time
        self.value = value
        self.error = error
        self.name = name
        self.original_data = {'time': self.time.copy(), 'value': self.value.copy(), 'error': self.error.copy()}
        self.FitLinear()
        self.original_trendline = self.trendline
        self.CaculateVarance()
        self.Duration = self.time[-1]-self.time[0]
        self.Bins = len(self.time)
    def Bin(self, tau):
        self.time = self.original_data['time'].copy()
        self.value = self.original_data['value'].copy()
        self.error = self.original_data['error'].copy()
        self.binning = {}
        nbins = 0
        tval = self.time[0]
        index = 0
        while tval < (self.time[0] + self.Duration):
            tval += tau
            nbins += 1
            index = self.FillBin(index, tval)
        self.time.clear()
        self.value.clear()
        self.error.clear()
        for time in self.binning.keys():
            self.time.append(time)
            self.value.append(self.binning[time][0])
            self.error.append(self.binning[time][1])
        del self.binning
        self.Bins = nbins
    def FillBin(self, last_index, bin_end):
        new_time_bin = []
        new_value_bin = []
        new_error_bin = []
        for i in range(last_index, len(self.time)):
            if self.time[i] >= bin_end:
                time = np.average(new_time_bin)
                value = np.average(new_value_bin)
                error = np.average(new_error_bin)
                self.binning[time] = (value, error)
                return i
            else:
                new_time_bin.append(self.time[i])
                new_value_bin.append(self.value[i])
                new_error_bin.append(self.error[i])
    def CaculateVarance(self):
        mu = np.average(self.value)
        diffs = []
        for i in range(0, len(self.value)):
            diffs.append(np.power(self.value[i] - mu - self.error[i], 2))
        self.sigma = np.average(diffs)
    def FitLinear(self):
        model = LinearRegression().fit(np.array(self.time).reshape(-1, 1), np.array(self.value))
        predicted = model.predict(np.array(self.time).reshape(-1, 1))
        self.trendline = predicted.tolist()
    def Detrend(self):
        self.FitLinear()
        self.value = (np.array(self.value) - np.array(self.trendline)).tolist()
    def Plot(self, original=True, detrended=False, trendline=False, fmt='.', save=False):
        if not original and not detrended:
            print('Nothing given to plot', file=sys.stderr)
            return
        fig = plt.figure(figsize=(10, 5))
        plot = fig.add_subplot(1, 1, 1)
        fig.suptitle('Time Series', fontsize=18)
        plot.set_xlabel(r'$Time$', fontsize=16)
        plot.set_ylabel(r'$\mathfrak{F}lux$', fontsize=16)
        if original:
            otime = self.original_data['time']
            ovalue = self.original_data['value']
            oerror = self.original_data['error']
            plt.errorbar(otime, ovalue, yerr=oerror, fmt=fmt, label='Original')
        if detrended:
            plt.errorbar(self.time, self.value, yerr=self.error, fmt=fmt, label='Detrended')
        if trendline:
            plt.plot(self.time, self.original_trendline, linewidth='0.7', color='r')
        plt.axhline(y=0, linestyle='-', linewidth='0.7', color='k')
        if original and detrended:
            plt.legend()
        if save:
            savename = '%s_ts.png' % self.name
            plt.savefig(savename)
        else:
            plt.show()

######################################################################################################################################

class StructureFunction:
    """
    Structure Function class
    """
    def __init__(self, time_series, name='structure_function'):
        self.time_series = time_series
        self.name = name
        self.duration = self.time_series.time[len(self.time_series.time)-1]-self.time_series.time[0]
        self.CalculateResolution()
        self.bins = int(self.duration/self.resolution)
        self.Initialize()
        self.calculated = False
        self.Calculate()
    def CalculateResolution(self):
        deltas = []
        for i in range(1, len(self.time_series.time)):
            deltas.append(self.time_series.time[i]-self.time_series.time[i-1])
        self.resolution = np.median(deltas)
    def Initialize(self):
        self.sf = {}
        self.raw_sf = {}
        self.raw_errors = {}
        for i in range(1, self.bins+1):
            midpoint = ((i-(1/2))*self.resolution)
            self.raw_sf[midpoint] = []
            self.raw_errors[midpoint] = []
    def Calculate(self):
        time = self.time_series.time
        value = self.time_series.value
        error = self.time_series.error
        N_e = len(time)
        for i in range(0, N_e - 1):
            for j in range(i, N_e):
                tau = time[j] - time[i]
                for btau in self.raw_sf.keys():
                    if tau >= (btau - (self.resolution/2)) and tau < (btau + (self.resolution/2)):
                        self.raw_sf[btau].append(np.power(value[j] - value[i], 2))
                        self.raw_errors[btau].append(error[i])
                        self.raw_errors[btau].append(error[j])
                        break
        self.BinRawSF()
        self.calculated = True
    def BinRawSF(self):
        taus = []
        values = []
        errors = []
        sigmas = []
        for tau in self.raw_sf.keys():
            sigmas.append(np.var(self.raw_errors[tau]))
        sigma_noise = 2.0 * np.average(sigmas)
        for tau in self.raw_sf.keys():
            if len(self.raw_sf[tau]) > 6:
                value = (np.average(self.raw_sf[tau] - sigma_noise)/self.time_series.sigma)
                sf_err = (np.sqrt(np.var(self.raw_sf[tau]))/np.sqrt(len(self.raw_sf[tau])/2))/self.time_series.sigma
                taus.append(tau)
                values.append(value)
                errors.append(sf_err)
        self.sf['tau'] = taus
        self.sf['sf'] = values
        self.sf['error'] = errors
    def ConvertToXspec(self):
        tau_start = []
        tau_stop = []
        taus = self.sf['tau'].copy()
        values = self.sf['sf'].copy()
        errors = self.sf['error'].copy()
        for i in range(0, len(taus) - 1):
            diff = (taus[i+1] - taus[i])/2
            tau_start.append(taus[i]-diff)
            tau_stop.append(taus[i]+diff)
        diff = (taus[len(taus)-1]-taus[len(taus)-2])/2
        tau_start.append(taus[len(taus)-1]-diff)
        tau_stop.append(taus[len(taus)-1]+diff)
        xsf = {'start':tau_start,'stop':tau_stop,'sf':values,'error':errors}
        return xsf
    def Plot(self, fmt='.', cutoff=None, save=False):
        if not self.calculated:
            print('Structure function not calculated, nothing to plot')
            return
        taus = self.sf['tau']
        values = self.sf['sf']
        errors = self.sf['error']
        if cutoff is not None:
            index = 0
            while taus[index] < cutoff:
                index += 1
                if index >= len(taus):
                    index = len(taus)
                    break
            taus = taus[:index-1]
            values = values[:index-1]
            errors = errors[:index-1]
        fig = plt.figure(figsize=(7, 10))
        plot = fig.add_subplot(1, 1, 1)
        fig.suptitle('Structure Function', fontsize=18)
        plot.set_xlabel(r'$\tau$', fontsize=16)
        plot.set_ylabel(r'$SF\left(\tau\right)$', fontsize=16)
        plt.errorbar(taus, values, yerr=errors, fmt=fmt, label='SF')
        plt.xscale('log')
        plt.yscale('log')
        if save:
            savename = '%s_sf.png' % self.name
            plt.savefig(savename)
        else:
            plt.show()

######################################################################################################################################


def calc_sf(t,x,x_err,LC_name = False, detrend = True, plot = False):
    
    
    ts = TimeSeries(time = t, value = x, error = x_err, name = LC_name)
    
    if detrend:
        ts.Detrend()
    
    sf = StructureFunction(time_series = ts, name = LC_name)
    
    if plot:
        # ts.Plot()
        sf.Plot()
        
    '''
    sf = StructureFunction(time_series = ts, name = LC_name) returns a "Structure Function object"
    
    need to access vectors within that object:
        sf_dict = sf.sf # this is a dict
        
        sf_dict.keys()
        Out[102]: dict_keys(['tau', 'sf', 'error'])
        
        
    '''

    return(sf)

######################################################################################################################################

def fit_sf(sf,tau_bend = False, plot = False, fit_each_point = True,LC_name = False):
    '''
    given a StructureFunction object, calculate the slope
    
    use optimize fitter (or something like that) to fit for the slope of the SF
    
    just be aware you'll need to scan the parameter space to make sure you're not 
    in a local minimum, just like you would use steppar in XSPEC to do
    
    sf = StructureFunction(time_series = ts, name = LC_name) returns a "Structure Function object"
    
    need to access vectors within that object:
        sf_dict = sf.sf # this is a dict
        
        sf_dict.keys()
        Out[102]: dict_keys(['tau', 'sf', 'error'])
    '''

    sf_dict = sf.sf # this is a dict
    
    tau = np.array(sf_dict['tau'])
    sf_vals = np.array(sf_dict['sf'])
    sf_err = np.array(sf_dict['error'])
    
    # remove negative SF values and their corresponding taus
    tau = tau[sf_vals > 0]
    sf_err = sf_err[sf_vals > 0]
    sf_vals = sf_vals[sf_vals > 0]
    
    if plot:
        plt.figure()
        plt.errorbar(x = tau, y = sf_vals, yerr = sf_err)
        plt.xscale('log');plt.yscale('log')
        plt.show()
    
    # get log of SF to fit
    logx = np.log10(tau)
    logy = np.log10(sf_vals)
    
    # remove values before bend
    if tau_bend:
        logy = logy[logx < np.log10(tau_bend)]
        logx = logx[logx < np.log10(tau_bend)]
    
    # fit for slope
    if fit_each_point == True:
        N = len(tau)
        fit = np.zeros(N)
        fit_err = np.zeros(N)
        for k in range(3,N):
            RESULTS = linregress(logx[0:k], logy[0:k], alternative='greater')
            fit[k] = RESULTS[0]
            fit_err[k] = RESULTS[4]
            RETURN = (tau,sf_vals,sf_err,fit,fit_err)
            
        plt.figure()
        if LC_name:
            plt.title(f'SF and fitted slope: {LC_name}')
        else:
            plt.title('SF and fitted slope')
        plt.errorbar(x = tau, y = fit + 1.0, yerr = fit_err, label = r'Correlation + 1 up to \n PSD slope $\sim$ SF slope + 1')
        plt.plot(tau,sf_vals,'k' ,label = 'SF')
        plt.xscale('log');plt.yscale('log')
        plt.xlabel(r'$\tau$')
        plt.legend()
        plt.tick_params(axis='y', which='minor')
        plt.grid()
        plt.show()
        
    if fit_each_point == False:
        RESULTS = linregress(logx, logy, alternative='greater')
        RETURN = (tau,sf_vals,sf_err,RESULTS)


    return(RETURN)



######################################################################################################################################

def calc_plane_of_sig(fn_sig,sig_levels,decay_const_multiplier = 1,tau = 'auto',
                      print_M_progress = False,custom_num_lc = False, WWZ_METHOD = 'Kirchner_numba',freq_method = 'log',freq=None):
    
    # fn_sig = '/Users/akshayghosh/wavelet_analysis/matlab_files/NGC6814_lc_ens/NGC_6814_swift_2022_ens_for_M_10lc.mat'

    # load matlab struct
    lc_data_struct = loadmat(fn_sig)
    Q = lc_data_struct; # open data structure

    # get info from data structure
    t = Q['time'][0];
    x_ens = Q['count_rate'];
    obs_title_vec = Q['name'][0]
    N = len(t)

    # initalize matrix for CWT coeffs
    if custom_num_lc:
        num_lc = custom_num_lc
    else:
        num_lc = np.shape(x_ens)[0]
    
    # num_lc = np.shape(x_ens)[0]

    # # calc sample W to get sizes
    # W_time_sample,W_freq_sample,W_sample = av.calc_wwz(t = t,x = x_ens[0],x_err = np.zeros(N),LC_name = False, 
    #                                                     plot = False,decay_const_multiplier = 1, freq_method_ = 'log',
    #                                                     num_levels_ = 100, v_percentiles = [0.1,97.5],calc_LSP = False,tau = np.linspace(0,np.max(t),len(t)))

    # if tau == 'auto':
    #     tau = np.linspace(0,np.max(t),len(t))
    # else:
    #     tau = None
    # decay_const_multiplier = 1
    default_decay_const = 1/(8*np.pi**2)


    res_sample = pyleo.utils.wavelet.wwz(ys = x_ens[0], ts = t, tau = tau, ntau=None, freq=freq, freq_method = freq_method, 
                            freq_kwargs={}, c = decay_const_multiplier*default_decay_const, Neff_threshold=3, 
                            Neff_coi=3, nproc=8, detrend=False, sg_kwargs=None, method=WWZ_METHOD, 
                            gaussianize=False, standardize=False, len_bd=0, bc_mode='reflect', reflect_type='odd')

    W_sample = res_sample.amplitude.T # wwz coeffs are real, not complex like the cwt?
    W_sample = np.square(W_sample)
    W_time_sample = res_sample.time
    COI_freq = 1/res_sample.coi

    s_T = np.shape(W_sample)
    # print('s_T: ',np.shape(s_T))
    W_ens = np.zeros([s_T[0] , s_T[1] , num_lc]) # create empty arrays for each output

    for i in tqdm(range(num_lc)):
    # for i in range(num_lc):
    #     if print_M_progress:
    #     #     pass
    #         print(i)
            # progress_bar(count_value = i, total = num_lc, suffix='')
        res = pyleo.utils.wavelet.wwz(ys = x_ens[i], ts = t, tau = tau, ntau=None, freq=freq, freq_method = freq_method, 
                                freq_kwargs={}, c = decay_const_multiplier*default_decay_const, Neff_threshold=3, 
                                Neff_coi=3, nproc=8, detrend=False, sg_kwargs=None, method=WWZ_METHOD, 
                                gaussianize=False, standardize=False, len_bd=0, bc_mode='reflect', reflect_type='odd')
        W_ens[:,:,i] = np.square(res.amplitude.T)
        '''
        % set values below coi = 0, for 3D
        %
        for k = 1:N
            T_f = T_ens(:,k,i);
            T_f(f < coi(k)) = 0;
            T_f(f > f_poisson) = 0; % also value above poisson noise = 0
            T_ens(:,k,i) = T_f;
        end
        %
        '''

    # calculate percentiles
    # sig_levels = 95
    M_amps = np.percentile(a = W_ens, q = sig_levels, axis = 2)
    
    return(M_amps)
    
######################################################################################################################################

def calc_wwz_obs(fn,fn_sig, sig_levels = [90,99], LC_name = False, save_fig_file = False, 
                 WWZ_METHOD = 'Kirchner_f2py', custom_num_lc = False):
    

        
    # fn = '/Users/akshayghosh/wavelet_analysis/matlab_files/qpe_paper_obs/qpe_paper_obs_csv/IRAS_13224-3809_rev_3044_300-5000_eV.csv'
    # fn_sig = '/Users/akshayghosh/wavelet_analysis/matlab_files/qpe_paper_lc_ensembles_test_1000/IRAS_13224-3809_rev_3044_300-5000_eV_ens_for_M.mat'
    
    # LC_name_ = 'iras 13224 rev 3044'
    
    x,t,bg,x_err,bg_err,obs_title = csv_to_lcdata(fn)
    
    # WWZ_METHOD_ = 'Kirchner_f2py'
    
    start = time.time()
    Q = calc_wwz(t,x,x_err,LC_name = LC_name, plot = True,
                 decay_const_multiplier = 1, freq_method_ = 'log',
                 num_levels_ = 100, v_percentiles = [0.1,97.5], 
                 calc_LSP = True,tau = None, 
                 plane_of_sig = True, fn_sig = fn_sig ,sig_levels = sig_levels, 
                 print_M_progress=True,custom_num_lc = custom_num_lc, WWZ_METHOD = WWZ_METHOD)
    end = time.time()
    
    delta_t = end - start
    
    print(f'delta t for WWZ_METHOD = {WWZ_METHOD}: {delta_t} s')
    
    return(Q)


######################################################################################################################################

def read_fits_suzaku(fn):
    
    lcfits = fits.open(fn)
    lcdata = lcfits[1].data
    
    t = lcdata.field('TIME')
    x = lcdata.field('RATE')
    x_err = lcdata.field('ERROR')

    return(t,x,x_err)

######################################################################################################################################

def find_powers_of_10(start, end):
    '''
    given a range of numbers, find all powers of 10 within that range
    eg: find_powers_of_10(12, 14532) = [100, 1000, 10000] 
    '''
    powers_of_10 = []
    power = math.floor(math.log10(start))

    while 10 ** power <= end:
        if 10 ** power >= start:
            powers_of_10.append(10 ** power)
        power += 1

    return(np.array(powers_of_10))

######################################################################################################################################

def read_fits_nustar(f,f_bg):
    # load data
    # FILENAME= fileThree_five
    lcfits = fits.open(f)
    lcdata = lcfits[1].data
    lcfits.close()
    timeThree_five  = lcdata.field('TIME')
    timeThree_five -= timeThree_five[0]
    rateThree_five  = lcdata.field('SUM12')
    errThree_five   = lcdata.field('SUM_E')
    # FILENAME= fileThree_fiveBK
    
    lcfits = fits.open(f_bg)
    lcdata = lcfits[1].data
    lcfits.close()
    timeThree_fiveBK  = lcdata.field('TIME')
    timeThree_fiveBK -= timeThree_fiveBK[0]
    rateThree_fiveBK  = lcdata.field('SUM12')
    errThree_fiveBK   = lcdata.field('SUM_E')
    
    # bkg subtract
    rateThree_fiveB = rateThree_five - rateThree_fiveBK
    errThree_fiveB   = np.sqrt(errThree_five**2 + errThree_fiveBK**2)
    
    t = timeThree_five
    x = rateThree_fiveB
    x_err = errThree_fiveB
    bg = rateThree_fiveBK
    bg_err = errThree_fiveBK
    
    # remove nans
    nan_indices = np.isnan(x) & np.isnan(x_err) & np.isnan(bg) & np.isnan(bg_err)
    t = t[~nan_indices]
    x = x[~nan_indices]
    x_err = x_err[~nan_indices]
    bg = bg[~nan_indices]
    bg_err = bg_err[~nan_indices]
    
    return(t,x,x_err,bg,bg_err)

######################################################################################################################################

def rebin_data(t, x, bin_size):
    '''
    improved binning function, to handle weird number of data points and NaN values
    '''
    
    # calculate the number of new bins
    num_new_bins = int(np.ceil(len(t) / bin_size))
    
    # calculate the new length of t and x based on the bin size
    new_len = num_new_bins * bin_size
    
    # pad t and x with zeros to make their lengths divisible by bin_size
    t_padded = np.append(t, np.zeros(new_len - len(t)))
    x_padded = np.append(x, np.zeros(new_len - len(x)))
    
    # reshape t and x into new bins
    # t_new = np.mean(t_padded[:new_len].reshape(num_new_bins, bin_size), axis=1)
    # x_new = np.mean(x_padded[:new_len].reshape(num_new_bins, bin_size), axis=1)
    t_new = np.nanmean(t_padded[:new_len].reshape(num_new_bins, bin_size), axis=1)
    x_new = np.nanmean(x_padded[:new_len].reshape(num_new_bins, bin_size), axis=1)
    
    return(t_new, x_new)


######################################################################################################################################

def calc_gravitational_radius_from_qpo(f_QPO,M):
    '''
    use eq 6 from Gallo,Blue, et all 2018 to calculate r from t_dyn (from f_QPO in Hz), and from r_g (from M)
    using a qpo freq
    
    return r in units of r_g
    
    paper:
    Eleven years of monitoring the Seyfert 1 Mrk 335 with Swift: Characterizing the X-ray and UV/optical variability
    doi:10.1093/mnras/sty1134
    '''
    # constants
    c = 299792458 # m/s
    G = 6.67e-11 # N*m^2 / kg^2
    M_sun = 1 # if M is in units of M_sun
    
    #r_g = G*M / (c**2)
    
    #r = np.power(((1/f_QPO) * np.power(3.7e-2 * 86400 * (M / (1e8*M_sun)) * np.power(r_g,-3.0/2), -1)),2.0/3)

    r = np.power(1 / (86400*f_QPO) , 2.0/3) * np.power(3.7e-2 * M / (1e8 * M_sun)    , -2.0/3)

    return(r)

######################################################################################################################################












