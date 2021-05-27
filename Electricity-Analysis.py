## By Noah Wilson
## 5/13/21
## This code performs a time-series analysis on hourly electric grid operating
## data in the US in order to try and determine the degree to which electric
## energy interchange between regions affects the patterns of electricity
## generation in those regions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

regions = ['CAL','NW','SW','CENT','TEX','MIDW','TEN','MIDA','NE','NY','SE','CAR','FLA']
api_key = 'f02c8bc09fa1f9f69d573d20cd9b4536'

## Start and end dates
## Format yyyymmddThhZ
## Can exclude hours, days or months
## Empty strings will give the entire available range

start ='20160101T00Z'
end = '20201231T23Z'

def get_data():
    ## This function gets the regional grid data from the EIA using API requests
    dfs = []
    for region in regions:
        api_call = api_call = f'http://api.eia.gov/series/?api_key={api_key}&start={start}&end={end}&series_id=EBA.{region}-ALL.NG.H'
        data = np.array(requests.get(api_call).json()['series'][0]['data'])
        timestamps = pd.to_datetime(data[:,0])
        Power = data[:,1].astype(np.float)
        dfs.append(pd.DataFrame({region:Power},index=timestamps))

    US_G_H = dfs[0].join(dfs[1:])
    US_G_H = US_G_H.dropna()
    return US_G_H

def get_coeff(Data):
    ## This function finds and determines the common Fourier coefficients for all regions

    com_coeff = 0
    coeff = pd.DataFrame([],columns=regions) # Dataframe with fourier coefficiencts
    for region in regions:
        a = np.fft.rfft(Data[region]-np.mean(Data[region])) # Fourier coefficients
        coeff[region] = a
        com_coeff += np.abs(a)/np.linalg.norm(a)# Normalized sum of fourier coefficients over regions
    return coeff,com_coeff

def decomp(C,com_C,Data,P):
    ## This code decomposes the net generation data and removes the trend and
    ## seasonal aspects, outputting the remainder for correlation analysis.

    b = np.percentile(abs(com_C),P) # value of the Pth percentile of coefficients
    idxs = np.where(abs(com_C)>=b)[0] # Indicies of coefficients above that percentile
    C.loc[idxs] = 0 # Set common frequency coefficients to zero
    Rt = pd.DataFrame([],columns=regions,index=Data.index) # Inverse ffts removing the common frequencies

    for region in regions:
        recon = np.fft.irfft(C[region],n=len(Data))
        Rt[region] = recon
    return Rt

def US48():

    ## This function creates the plots for the US48 data used as a test bench for the analytical process

    api_call = f'http://api.eia.gov/series/?api_key={api_key}&start={start}&end={end}&series_id=EBA.US48-ALL.NG.H'
    data = np.array(requests.get(api_call).json()['series'][0]['data'])
    timestamps = pd.to_datetime(data[:,0])
    Power = data[:,1].astype(np.float)
    Net_Gen = pd.DataFrame({'US48':Power},index=timestamps)
    Net_Gen = Net_Gen.dropna()

    ## These are the raw data plots
    fig,axs=plt.subplots(3,1,figsize=[12,6])
    axs[0].plot(Net_Gen['US48'])
    axs[1].plot(Net_Gen.loc['20201231T23Z':'20200101T00Z'])
    axs[2].plot(Net_Gen.loc['20200131T23Z':'20200101T00Z'])
    axs[2].set_xlabel('Date')
    axs[1].set_ylabel('Net Generation (MWh)')

    ## FFT of the US48 data
    plt.figure()
    ax = plt.gca()
    a = abs(np.fft.rfft((Net_Gen-np.mean(Net_Gen))['US48'])) # Fourier coefficients
    ax.plot(np.fft.rfftfreq(len(Net_Gen)),a)
    ax.set_xlabel('Freqency (Hz)')

    ## Time-series decomposition of one month of US48 data
    Y = Net_Gen.loc['20200131T23Z':'20200101T00Z']
    a = abs(np.fft.rfft((Y-np.mean(Y))['US48'])) # Fourier coefficients
    b = np.percentile(abs(a),99)
    a[abs(a)<b] = 0
    S = np.fft.irfft(a)
    R = Y['US48']-np.mean(Y['US48']) - S

    fig1,axs1=plt.subplots(3,1,figsize=[12,6],sharex=True)
    axs1[0].plot(Y.index,Y['US48'])
    axs1[0].set_ylabel(r'$X_t$')
    axs1[1].plot(Y.index,S)
    axs1[1].set_ylabel(r'$S_t - T_t$')
    axs1[2].plot(Y.index,R)
    axs1[2].set_ylabel(r'$R_t$')
    axs1[2].set_xlabel('Date')
    plt.show()

def regional():
    ## This function performs the regional analysis and outputs all of the necessary plots

    US_G_H = get_data() # Creates dataframe for the hourly generation data
    Tt = np.mean(US_G_H) # "Trend" of the time-series data (Just the mean)
    coeff,com_coeff = get_coeff(US_G_H)

    ## Plots the common frequencies of the FFTs
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(np.fft.rfftfreq(len(US_G_H)),com_coeff)
    ax1.set_xlabel('Frequency (Hz)')

    ## Creates a plot for how the norm of the correlation matrix changes
    ## with changing percentile threshold.
    normr = []
    for per in -np.arange(-100,1):
        Rt = decomp(coeff,com_coeff,US_G_H,per)
        X = np.array(Rt/np.linalg.norm(Rt,axis=0)).T
        r = X@X.T # Correlation matrix for all regions
        normr.append(np.linalg.norm(r))

    plt.figure()
    ax2 = plt.gca()
    ax2.plot(-np.arange(-100,1),normr,'o')
    ax2.set_xlabel('Percentile Threshold')
    ax2.set_ylabel(r'$||R_t||_2$')

    ## Plots the correlation matrix for the raw generation data, with no
    ## trend or seasonality adjustments
    plt.figure(figsize=(7,6))
    ax3 = plt.gca()
    X = np.array(US_G_H/np.linalg.norm(US_G_H,axis=0)).T
    R = X@X.T
    im = ax2.matshow(X@X.T,vmin=0,vmax=1) # Plot the correlation matrix
    ax3.set_xticks(np.arange(0,13))
    ax3.set_yticks(np.arange(0,13))
    ax3.set_xticklabels(regions,rotation=45)
    ax3.set_yticklabels(regions)
    plt.colorbar(im,ax=ax3)

    ## Plots the correlation matrices of the detrended and deseasoned generation data
    ## for three different chosen percentile thresholds.
    fig,axs=plt.subplots(1,3,figsize=[15,4],sharey=True)
    for i,P in enumerate([99,89,63]):

        Rt = decomp(coeff,com_coeff,US_G_H,P)
        Xt = np.array(Rt/np.linalg.norm(Rt,axis=0)).T
        R = Xt @ Xt.T
        im = axs[i].matshow(R,vmin=0,vmax=1)
        axs[i].set_xticks(np.arange(0,13))
        axs[i].set_yticks(np.arange(0,13))
        axs[i].set_xticklabels(regions,rotation=90)
        axs[i].set_yticklabels(regions)
        axs[i].set_title(f'Threshold: {P}%',y=-0.25)
    fig.colorbar(im, ax=axs.ravel().tolist())

    plt.show()

if __name__ == '__main__':
    ## Uncomment US48() to get the plots for the entire continental united states.
    ## Leave it commented to get only the regional analysis plots.

    # US48()
    regional()
