# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:26:57 2023

Scatter the Pandora & surface data - all locations on their own plots
All available dates & locations combined

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D
import math

pollutants = ['HCHO','NO2','O3','SO2']
dtype = ['surface','surface','column','column']
typeLabel = ['Surface','Surface','Column','Column']

#use interquartile range for Pandora instead of full range?
IQR = 'yes'

for n in range(len(pollutants)):
    
    if pollutants[n] == 'O3':
        #get the relevant location data for each
        locations = ['Bayonne','Bristol','CapeElizabeth','Cornwall','EastProvidence','Londonderry','Lynn','MadisonCT','NewBrunswick','NewHaven','OldField','Philadelphia','Pittsburgh','Queens','WashingtonDC','Westport','AFRC','TMF','Ames','Richmond','CSUS','SLC','BAO','NREL','Platteville','AldineTX','LibertyTX','HoustonTX']
        pods = ['EPA_Bayonne','EPA_Bristol','EPA_CapeElizabeth','EPA_Cornwall','EPA_EastProvidence','EPA_Londonderry','EPA_Lynn','EPA_MadisonCT','EPA_NewBrunswick','EPA_NewHaven','EPA_OldField','EPA_Philadelphia','EPA_Pittsburgh','EPA_Queens','EPA_WashingtonDC','EPA_Westport','YPODR9','YPODA2','YPODL6','YPODL1','YPODL2','WBB','BAO_ground','NREL_ground','Platteville_ground','HoustonAldine','LibertySamHoustonLibrary','UHMoodyTower']
        colors = plt.get_cmap('tab20').colors + plt.get_cmap('tab10').colors[:8]

    elif pollutants[n] == 'NO2':
        #get the relevant location data for each
        locations = ['NewHaven','Westport','WashingtonDC','Lynn','Londonderry','Bayonne','NewBrunswick','OldField','Queens','Philadelphia','Pittsburgh','EastProvidence','Hawthorne','Hampton','SWDetroit','AldineTX']
        pods = ['EPA_NewHaven','EPA_Westport','EPA_WashingtonDC','EPA_Lynn','EPA_Londonderry','EPA_Bayonne','EPA_NewBrunswick','EPA_OldField','EPA_Queens','EPA_Philadelphia','EPA_Pittsburgh','EPA_EastProvidence','EPA_Hawthorne','EPA_Hampton','EPA_Detroit','EPA_Aldine']
        colors = plt.get_cmap('tab20').colors[:len(locations)]

    elif pollutants[n] == 'SO2':
        #get the relevant location data for each
        locations = ['NewHaven','WashingtonDC','Londonderry','Bayonne','Bronx','Queens','Philadelphia','Pittsburgh','EastProvidence','Hawthorne','SWDetroit']
        pods = ['EPA_NewHaven','EPA_WashingtonDC','EPA_Londonderry','EPA_Bayonne','EPA_Bronx','EPA_Queens','EPA_Philadelphia','EPA_Pittsburgh','EPA_EastProvidence','EPA_Hawthorne','EPA_Detroit']
        colors = plt.get_cmap('tab20').colors[:len(locations)]
        
    elif pollutants[n] == 'HCHO': #CA pods only
        locations = ['Ames','CSUS','Richmond','Whittier','TMF','AFRC']
        pods = ['YPODL6','YPODL2','YPODL1','YPODA7','YPODA2','YPODR9']
        colors = plt.get_cmap('tab10').colors[:len(locations)]
        
    #Create subplots
    fig, axs = plt.subplots(nrows=math.ceil(len(locations)/4), ncols=4, figsize=(15, 5 * math.ceil(len(locations)/4)))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten to make indexing easier

    for k in range(len(locations)):
        #-------------------------------------
        #initialize our counter for how many samples we have
        num = 0
    
        #-------------------------------------
        #first load in the pandora csv's
        pandoraPath = 'C:\\Users\\kokorn\\Documents\\Modeling Surface Concentrations\\Pandora'
        #get the filename for pandora
        pandorafilename = "{}_{}_extra_{}.csv".format(locations[k],dtype[n],pollutants[n])
        #join the path and filename
        pandorafilepath = os.path.join(pandoraPath, pandorafilename)
        #different load structure for FRAPPE data
        if locations[k] == 'BAO' or locations[k] == 'NREL' or locations[k] == 'Platteville':
           pandora = pd.read_csv(pandorafilepath,index_col=0)
        else:
           pandora = pd.read_csv(pandorafilepath,index_col=1)
        #Convert the index to a DatetimeIndex and set the nanosecond values to zero
        pandora.index = pd.to_datetime(pandora.index)
        #Filter so that the lowest quality data is NOT included
        if locations[k] != 'BAO' and locations[k] != 'NREL' and locations[k] != 'Platteville':
           pandora = pandora.loc[pandora['quality_flag'] != 12]
        #if surface, convert mol/m3 to ppb
        if typeLabel[n] == 'Surface': #if not surface, cannot convert
            pandora['{}'.format(pollutants[n])] = pandora['{}'.format(pollutants[n])]*0.08206*pandora['temperature']*(10**(9))/1000
        #get rid of any unnecessary columns
        pandora = pandora[['{}'.format(pollutants[n])]]
        #resample to hourly to match ground data
        pandora = pandora.resample('H').mean()    
        #Change the pollutant column name
        pandora.columns.values[0] = 'Pandora {} {}'.format(typeLabel[n],pollutants[n])
        #remove any negatives
        pandora = pandora[pandora.iloc[:, 0] >= 0]
        #-------------------------------------
        
        #now load in the matching ground data
        podPath = 'C:\\Users\\kokorn\\Documents\\Modeling Surface Concentrations\\Ground'
        #get the filename for the pod
        podfilename = "{}_{}.csv".format(pods[k],pollutants[n])
        #read in the first worksheet from the workbook myexcel.xlsx
        podfilepath = os.path.join(podPath, podfilename)
        
        if 'WBB' in pods[k]: #wbb, slc
            pod = pd.read_csv(podfilepath,skiprows=10,index_col=1)
            #delete the first row - holds unit info
            pod = pod.drop(pod.index[0])
        else:
            #all others take the same format
            pod = pd.read_csv(podfilepath,index_col=0)  
        
        #Convert Index to DatetimeIndex
        if 'YPOD' in pods[k]:
            pod.index = pd.to_datetime(pod.index, format="%d-%b-%Y %H:%M:%S")
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            pod.index = pd.to_datetime(pod.index.values.astype('datetime64[s]'), errors='coerce')
            #if not FRAPPE, need to change the column names
            if locations[k] != 'BAO' or locations[k] != 'NREL' or locations[k] != 'Platteville':
                pod.rename(columns={'Y_hatfield':'{}'.format(pollutants[n])}, inplace=True)
        
        #if it's a non-pod, need to clean the data more
        elif 'topaz' in pods[k]: #topaz, noaa
            #need different format for non-pods
            pod.index = pd.to_datetime(pod.index, format="%Y-%m-%d %H:%M:%S")
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            pod.index = pd.to_datetime(pod.index.values.astype('datetime64[s]'), errors='coerce')
            #Drop all columns except the specified one
            columns_to_drop = [col for col in pod.columns if col != 'O3_1m,ppbv']
            pod.drop(columns=columns_to_drop, inplace=True)
            #rename the main column
            pod.rename(columns={'O3_1m,ppbv':'O3'}, inplace=True)
            #remove rows containing -999
            pod = pod[pod['O3'] != -999]
        
        elif 'WBB' in pods[k]: #wbb, slc
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            pod.index = pd.to_datetime(pod.index.values.astype('datetime64[s]'), errors='coerce')
            #Drop all columns except the specified one
            pod = pod[['ozone_concentration_set_1']]#rename the main column
            #rename the ozone column
            pod.rename(columns={'ozone_concentration_set_1':'O3'}, inplace=True)
            #remove rows containing -999
            pod = pod[pod['O3'] != -999]
            #make sure our o3 data is numbers, not strings
            pod['O3'] = pod['O3'].astype(float)

        elif 'TX' in locations[k]: #texas data
            pod.index = pd.to_datetime(pod.index)
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            pod.index = pd.to_datetime(pod.index.values.astype('datetime64[s]'), errors='coerce')
        
        elif 'EPA' in pods[k]: #EPA data
            pod.index = pd.to_datetime(pod.index)
            
        elif 'ground' in pods[k]: #FRAPPE ref data
            pod.index = pd.to_datetime(pod.index)
            #remove whitespace from column headings
            pod.columns = pod.columns.str.strip()
            #remove rows containing -999
            pod = pod[pod['{}'.format(pollutants[n])] != -999]
        
        #resample to hourly
        pod = pod.resample('H').mean()
    
        #-------------------------------------
        #merge our dataframes
        merge = pd.merge(pandora,pod,left_index=True, right_index=True)
        #remove missing values for ease of plotting
        merge = merge.dropna()
        
        #-------------------------------------
        #limit Pandora measurements to their IQR 
        if IQR == 'yes':
           #Calculate the interquartile range (IQR) for the tropo pandora
           q1 = merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])].quantile(0.25)
           q3 = merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])].quantile(0.75)
           iqr = q3 - q1
       
           #Set the y-limits based on the IQR
           x_min = q1 - 1.5 * iqr
           x_max = q3 + 1.5 * iqr
       
           #filter the dataframe based on the IQR
           merge = merge[(merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])] >= x_min) & (merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])] <= x_max)]
           
           #-------------------------------------------------------------
           # Calculate the interquartile range (IQR) for the surface pandora
           q1 = merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])].quantile(0.25)
           q3 = merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])].quantile(0.75)
           iqr = q3 - q1
       
           #Set the y-limits based on the IQR
           x_min = q1 - 1.5 * iqr
           x_max = q3 + 1.5 * iqr
       
           #filter the dataframe based on the IQR
           merge = merge[(merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])] >= x_min) & (merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])] <= x_max)]
    
        #-------------------------------------
        #add the num of measurements to our counter
        num = len(merge)
        #add the new data to our scatterplot
        axs[k].scatter(merge['{}'.format(pollutants[n])], merge['Pandora {} {}'.format(typeLabel[n],pollutants[n])],c=colors[k], s=25)

        #Final touches for plotting
        #Add subplot title with the location
        axs[k].set_title(f'{locations[k]}')
        #Add x and y axis labels
        axs[k].set_xlabel('Ground {} (ppb)'.format(pollutants[n]))
        if typeLabel[n] == 'Surface':
            axs[k].set_ylabel('Pandora {} {} (ppb)'.format(typeLabel[n],pollutants[n]))
        else:
            axs[k].set_ylabel('Pandora {} {} (mol/m2)'.format(typeLabel[n],pollutants[n]))
        #Add text in different colors
        axs[k].text(0.525, 0.93, 'n = {}'.format(num), fontsize=12, color='black', transform=axs[k].transAxes)

        #Adding a title to fig4
        if IQR == 'yes':
            fig.suptitle('{} - Pandora IQR vs. Ground {}'.format(typeLabel[n],pollutants[n]), y=.93)  # Adjust the vertical position (0 to 1)
        else:
            fig.suptitle('{} - Pandora vs. Ground {}'.format(typeLabel[n],pollutants[n]), y=.93)  # Adjust the vertical position (0 to 1)
        #Add a 1:1 line
        axs[k].plot([axs[k].get_xlim(), axs[k].get_ylim()], [axs[k].get_xlim(), axs[k].get_ylim()], color='black', linestyle='--', label='1:1 Line')
   
    # Hide any unused subplots
    for ax in axs[len(locations):]:
        ax.set_visible(False)
    
    #Display the plot
    plt.show()
    
    #save to a different folder so we don't confuse the script on the next iteration
    Spath = 'C:\\Users\\kokorn\\Documents\\Modeling Surface Concentrations\\Plain Scatterplots\\'
    #Create the full path with the figure name
    if IQR == 'yes':
        savePath = os.path.join(Spath,'PandoraIQR_ground_scatter_{}_subplots'.format(pollutants[n]))
    else:
        savePath = os.path.join(Spath,'Pandora_ground_scatter_{}_subplots'.format(pollutants[n]))
    #Save the figure to a filepath
    fig.savefig(savePath)