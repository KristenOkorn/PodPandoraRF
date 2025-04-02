# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:58:58 2024

Load & reformat the Texas AQ data

Access data here: https://www.tceq.texas.gov/cgi-bin/compliance/monops/yearly_summary.pl

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import os
from datetime import datetime, timedelta

#loop through locations & pollutants
locations = ['HoustonAldine','LibertySamHoustonLibrary','StEdwards','UHLaunchTrailer','UHMoodyTower','WacoManazec']
pollutants = ['O3']

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\2023 Deployment\\Modeling Surface Concentrations\Texas'
 
for k in range(len(locations)):
    for n in range(len(pollutants)):
            #generate filename & path
            filename = '{}_{}.txt'.format(locations[k],pollutants[n])
            filepath = os.path.join(path, filename)
            
            #open the file while skipping the headers
            texas = pd.read_csv(filepath,skiprows=4,header=1,delimiter=',')
            
            #Melt the DataFrame to combine date and time into a single column and values into another column
            texas = pd.melt(texas, id_vars=['Date'], var_name='Hour', value_name='{}'.format(pollutants[n]))

            #Combine 'Date' and 'Hour' into a single column 'DateTime'
            texas['datetime'] = texas['Date'] + ' ' + texas['Hour']

            # Convert to datetime format
            texas['datetime'] = pd.to_datetime(texas['datetime'])
            
            #convert to UTC (to match pandora)
            texas['datetime'] = texas['datetime'] - timedelta(hours=6)
            
            #re-order to ascending order
            texas = texas.sort_values(by='datetime')

            #drop extra columns
            texas = texas.drop(['Date', 'Hour'], axis=1)

            #Set the datetime as the index
            texas.set_index('datetime', inplace=True)
            
            #Replace strings with NaN and convert the rest to float
            texas['{}'.format(pollutants[n])] = pd.to_numeric(texas['{}'.format(pollutants[n])], errors='coerce')
     
            #save out the final data
            savePath = os.path.join(path,'{}_{}.csv'.format(locations[k],pollutants[n]))
            texas.to_csv(savePath)