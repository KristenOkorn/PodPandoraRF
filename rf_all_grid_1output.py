# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 03:36:52 2024

Editing 7/18/24 to save outputs from each run to 1 large file

Edited 7/5/2024 - adding SO2 & parallelization

Runs separately for all matching surface-Pandora pairs

Using grid search for hyperparameter tuning


@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import multiprocessing

#which pollutant to model
pollutant = 'O3'
#datatype will vary based on pollutant
dtype = 'column'
#use bagging?
bagging = 'no'

if pollutant == 'O3':
    #list of pods to model
    locations = ['Bayonne','Bristol','CapeElizabeth','Cornwall','EastProvidence','Londonderry','Lynn','MadisonCT','NewBrunswick','NewHaven','OldField','Philadelphia','Pittsburgh','Queens','WashingtonDC','Westport','AFRC','TMF','Ames','Richmond','CSUS','SLC','BAO','NREL','Platteville','AldineTX','LibertyTX','HoustonTX']
    pods = ['EPA_Bayonne','EPA_Bristol','EPA_CapeElizabeth','EPA_Cornwall','EPA_EastProvidence','EPA_Londonderry','EPA_Lynn','EPA_MadisonCT','EPA_NewBrunswick','EPA_NewHaven','EPA_OldField','EPA_Philadelphia','EPA_Pittsburgh','EPA_Queens','EPA_WashingtonDC','EPA_Westport','YPODR9','YPODA2','YPODL6','YPODL1','YPODL2','WBB','BAO_ground','NREL_ground','Platteville_ground','HoustonAldine','LibertySamHoustonLibrary','UHMoodyTower']
    #smaller subset for troubleshooting
    #locations = ['BAO','NREL']
    #pods = ['BAO_ground','NREL_ground']
    
    #create a directory path for us to pull from / save to
    path = 'C:\\Users\\kokorn\\Documents\\Modeling Surface Concentrations\\All O3 Data Combined'
    
elif pollutant == 'SO2':
    #get the relevant location data for each
    locations = ['NewHaven','WashingtonDC','Londonderry','Bayonne','Bronx','Queens','Philadelphia','Pittsburgh','EastProvidence','Hawthorne','SWDetroit']
    pods = ['EPA_NewHaven','EPA_WashingtonDC','EPA_Londonderry','EPA_Bayonne','EPA_Bronx','EPA_Queens','EPA_Philadelphia','EPA_Pittsburgh','EPA_EastProvidence','EPA_Hawthorne','EPA_Detroit']
    
    #create a directory path for us to pull from / save to
    path = 'C:\\Users\\kokorn\\Documents\\Modeling Surface Concentrations\\All SO2 Data Combined'
    
#Make a path to save our outputs to later
if bagging == 'yes':    
    #Name a new subfolder
    subfolder_name = 'Outputs_{}_rf_indiv_bag_grid'.format(pollutant)
else:
    subfolder_name = 'Outputs_{}_rf_indiv_grid'.format(pollutant)
    
#Create the subfolder path
subfolder_path = os.path.join(path, subfolder_name)
#Create the subfolder
os.makedirs(subfolder_path, exist_ok=True)

#----------------------------------------------------------
#make a function that loads our data and fits the rf algorithm
def load_n_fit(n, pollutant, dtype, bagging, location, pod, path):
    
    #load the Pandora (ML input) data
    pandorafilename = "{}_{}_extra_{}.csv".format(location,dtype,pollutant)
    #combine the file and the path
    pandorafilepath = os.path.join(path, pandorafilename)
    #different load structure for FRAPPE data
    if location == 'BAO' or location == 'NREL' or location == 'Platteville':
        ann_inputs = pd.read_csv(pandorafilepath,index_col=0)
    else:
        ann_inputs = pd.read_csv(pandorafilepath,index_col=1)
    #Convert the index to a DatetimeIndex and set the nanosecond values to zero
    ann_inputs.index = pd.to_datetime(ann_inputs.index)
    #remove any empty/placeholder times
    if pollutant == 'O3' and location != 'BAO' and location != 'NREL' and location != 'Platteville': 
        ann_inputs = ann_inputs[ann_inputs['Atmos Variability'] != 999]
        ann_inputs = ann_inputs[ann_inputs['O3'] != -9e99]
    
    #get rid of unnecessary FRAPPE columns
    if location == 'BAO' or location == 'NREL' or location == 'Platteville':
        ann_inputs = ann_inputs.rename(columns={'Temp-Eff': 'TEff','AMF':'O3 AMF'})
        #for FRAPPE pods, also need to add dummy columns
        ann_inputs['Atmos Variability'] = 3
        ann_inputs['O3 AMF'] = 2
        #Define the desired order of columns
        desired_order = ['SZA', 'pressure', 'O3', 'TEff', 'O3 AMF', 'Atmos Variability']
        #re-order the columns
        ann_inputs = ann_inputs[desired_order]
        #remove any nans from retime
        ann_inputs = ann_inputs.dropna()
    else: #others don't need this
        #Filter so that the lowest quality data is NOT included
        ann_inputs = ann_inputs.loc[ann_inputs['quality_flag'] != 12]
        if pollutant == 'O3':
            ann_inputs = ann_inputs[['SZA','pressure','{}'.format(pollutant),'TEff','{} AMF'.format(pollutant),'Atmos Variability']]
        else:
            ann_inputs = ann_inputs[['SZA','pressure','{}'.format(pollutant),'TEff','AMF','Atmos Variability']]
        #data cleaning - remove negatives from the pollutant column
    ann_inputs.loc[ann_inputs['{}'.format(pollutant)] < 0, '{}'.format(pollutant)] = np.nan

    #--------------------------------------------
    #now load the matching ground data
    groundfilename = "{}_{}.csv".format(pod,pollutant)
    #combine the file and the path
    groundfilepath = os.path.join(path, groundfilename)
    if 'WBB' in pod: #wbb, slc
        ground = pd.read_csv(groundfilepath,skiprows=10,index_col=1)
        #delete the first row - holds unit info
        ground = ground.drop(ground.index[0])
    else:
        #all others take the same format
        ground = pd.read_csv(groundfilepath,index_col=0)  
    #Convert Index to DatetimeIndex
    if 'YPOD' in pod:
        ground.index = pd.to_datetime(ground.index, format="%d-%b-%Y %H:%M:%S")
        #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
        ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
        #if FRAPPE, need to change the column names
        if location == 'BAO' or location == 'NREL' or location == 'Platteville':
            ground.rename(columns={'O3':'Y_hatfield'}, inplace=True)
    
    #if it's a non-pod, need to clean the data more
    elif 'topaz' in pod: #topaz, noaa
        #need different format for non-pods
        ground.index = pd.to_datetime(ground.index, format="%Y-%m-%d %H:%M:%S")
        #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
        ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
        #Drop all columns except the specified one
        columns_to_drop = [col for col in ground.columns if col != 'O3_1m,ppbv']
        ground.drop(columns=columns_to_drop, inplace=True)
        #rename the main column
        ground.rename(columns={'O3_1m,ppbv':'Y_hatfield'}, inplace=True)
        #remove rows containing -999
        ground = ground[ground['Y_hatfield'] != -999]
    
    elif 'WBB' in pod: #wbb, slc
        #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
        ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
        #Drop all columns except the specified one
        ground = ground[['ozone_concentration_set_1']]#rename the main column
        #rename the ozone column
        ground.rename(columns={'ozone_concentration_set_1':'Y_hatfield'}, inplace=True)
        #remove rows containing -999
        ground = ground[ground['Y_hatfield'] != -999]
        #make sure our o3 data is numbers, not strings
        ground['Y_hatfield'] = ground['Y_hatfield'].astype(float)

    elif 'TX' in location: #texas data
        #this one has TX in locations - all others have signifier in 'pods'
        ground.rename(columns={'O3':'Y_hatfield'}, inplace=True)
        ground.index = pd.to_datetime(ground.index, format="%Y-%m-%d %H:%M:%S")
        #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
        ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
    
    elif 'EPA' in pod: #EPA data
        ground.rename(columns={'{}'.format(pollutant):'Y_hatfield'}, inplace=True)
        ground.index = pd.to_datetime(ground.index)
        #convert from ppm to ppb
        ground['Y_hatfield'] = ground['Y_hatfield'] * 1000
        
    elif 'ground' in pod: #FRAPPE ref data
        ground.index = pd.to_datetime(ground.index)
        #remove whitespace from column headings
        ground.columns = ground.columns.str.strip()
        #remove rows containing -999
        ground = ground[ground['{}'.format(pollutant)] != -999]
        ground.rename(columns={'{}'.format(pollutant):'Y_hatfield'}, inplace=True)
        
    #let us know we've finished loading
    print(f"{pollutant} data from {location}: Data loaded")
   
    #---------------------------------------------------
    #remove any nans before retime
    ground = ground.dropna()
    #resample to hourly 
    ground = ground.resample('H').mean()
    
    #combine our datasets - both already in local time
    x=pd.merge(ann_inputs,ground,left_index=True,right_index=True)
    #remove NaNs
    x = x.dropna()
    
    #Remove whitespace from column labels
    x.columns = x.columns.str.strip()
    
    #now for reformatting - get our 'y' data alone
    y = pd.DataFrame(x.pop('Y_hatfield'))
    
    #Now do our test-train split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    
    #scale our dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Create a Random Forest regressor object
    rf_regressor = RandomForestRegressor()
    
    #Define a parameter grid to search over
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [6,7,8,9,10], #Hiro recommended
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        }
    
    if bagging == 'no':
        #initialize the GridSearchCV object
        grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

    elif bagging == 'yes':
        #Create a Bagging Regressor with the base Random Forest Regressor
        bagging_rf_model = BaggingRegressor(rf_regressor, random_state=42)
        #Create a pipeline with a scaler and the bagging regressor
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('bagging', bagging_rf_model)
            ])
        
        #Update the parameter grid for the pipeline
        param_grid_pipeline = {
            'bagging__base_estimator__n_estimators': [50, 100],
            'bagging__base_estimator__max_depth': [None, 10, 20],
            'bagging__base_estimator__min_samples_split': [2, 5, 10],
            'bagging__base_estimator__min_samples_leaf': [1, 2, 4],
            'bagging__base_estimator__max_features': ['auto', 'sqrt', 'log2'],
            'bagging__n_estimators': [10, 20],  # Number of base estimators in the ensemble
            'bagging__bootstrap_features': [True, False],
            }
        #Create the GridSearchCV object with BaggingRegressor
        grid_search = GridSearchCV(pipeline, param_grid_pipeline, cv=5, scoring='neg_mean_squared_error')
    
    #---------------------------------------------------
    #Fit the model to the data
    grid_search.fit(X_train_scaled, y_train['Y_hatfield'].values)

    #let us know we've finished fitting the data
    print(f"{pollutant} data from {location}: Model fitted")
    
    #get the best parameters found
    best_params = grid_search.best_params_
    #convert the dictionary to a pandas DataFrame
    best_params = pd.DataFrame.from_dict(best_params,orient='index')
    #convert back to a dictionary with the location identifier
    best_params = {'AppLoc': location, 'data': best_params}
    
    #get the best model
    best_rf_model = grid_search.best_estimator_
        
    #make predictions
    y_hat_train = best_rf_model.predict(X_train_scaled)
    y_hat_test = best_rf_model.predict(X_test_scaled)
    
    #now add the predictions  & y's back to the original dataframes
    X_train['y_hat_train'] = y_hat_train
    X_test['y_hat_test'] = y_hat_test
    X_train['Y'] = y_train
    X_test['Y'] = y_test
    
    if bagging == 'no':
        #Get the feature importances
        importances = best_rf_model.feature_importances_
        #Create a dictionary of importance labels with their corresponding input labels
        importance_labels = {label: importance for label, importance in zip(x.columns, importances)}
        #convert to dataframe
        importance_labels = pd.DataFrame(importance_labels)
        #save out separately for each location
        savePath = os.path.join(subfolder_path,'{}_importancelabels_{}.csv'.format(location,pollutant))
        importance_labels.to_csv(savePath,index=False)
    #---------------------------------------------------
                
    #generate statistics for test data
    r2 = r2_score(y_test['Y_hatfield'], y_hat_test)
    rmse = np.sqrt(mean_squared_error(y_test['Y_hatfield'], y_hat_test))
    mbe = np.mean(y_hat_test - y_test['Y_hatfield'])
    #store our results in a dictionary
    stats_test = {'AppLoc': location,'R2': r2, 'RMSE': rmse, 'MBE': mbe}
    #append these to the main list
    #stats_test_list.append(stats_test)
    
    #generate statistics for train data
    r2 = r2_score(y_train['Y_hatfield'], y_hat_train)
    rmse = np.sqrt(mean_squared_error(y_train['Y_hatfield'], y_hat_train))
    mbe = np.mean(y_hat_train - y_train['Y_hatfield'])
    #store our results in a dictionary
    stats_train = {'AppLoc': location,'R2': r2, 'RMSE': rmse, 'MBE': mbe}
    #append these to the main list
    #stats_train_list.append(stats_train)
    
    #save the best rf model
    savePath = os.path.join(subfolder_path,'{}_rfmodel_{}.joblib'.format(location,pollutant))
    dump(best_rf_model, savePath)
    
    #--------------------------------------------------- 
    result = {
        'best_params': best_params,
        'stats_test': stats_test,
        'stats_train': stats_train
        }
    return result
        
#---------------------------------------------------
#Function to process each pair of location and pod
def process_location_pod(location_pod):
    location, pod = location_pod
    return load_n_fit(pollutant, dtype, bagging, location, pod, path, subfolder_path)

#Add these lines for multiprocessing on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

    #Pair locations and pods
    location_pod_pairs = list(zip(locations, pods))

    #Use ProcessPoolExecutor to parallelize
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #Map the function to the list of location_pod_pairs to run in parallel
        results = list(executor.map(process_location_pod, location_pod_pairs))
        
    #create a list to store the stats for each model for this pollutant
    stats_test_list = []
    stats_train_list = []
    best_params_list = []
    
    #---------------------------------------------------
    for result in results: #save the stats from each run 
        best_params_list.append(result['best_params'])
        stats_test_list.append(result['stats_test'])
        stats_train_list.append(result['stats_train'])

    # Convert lists of dictionaries to DataFrames
    to_save = ['best_params', 'stats_test', 'stats_train']
    dataframes = [best_params_list, stats_test_list, stats_train_list]  # List of actual data

    for i, data in enumerate(dataframes):
        df = pd.DataFrame(data)
        savePath = os.path.join(subfolder_path, '{}_{}.csv'.format(to_save[i], pollutant))
        df.to_csv(savePath, index=False)

    print("Statistics saved to CSV files.")