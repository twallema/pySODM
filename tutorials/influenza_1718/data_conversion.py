############################
## Load required packages ##
############################

import os
import datetime
import pandas as pd

###############
## Load data ##
###############

abs_dir = os.path.dirname('')
rel_dir = 'data/raw/dataset_influenza_1718.csv'
data_dir = os.path.join(abs_dir,rel_dir)
data = pd.read_csv(data_dir)
# Age groups in dataset
desired_age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)],closed='left')
# Hardcode Belgian demographics (Jan 1, 2018)
N = pd.Series(index=desired_age_groups, data=[620914, 1306826, 7317774, 2130556])

########################
## Perform conversion ##
########################

# Convert YEAR-WEEK to a date
data = pd.read_csv(data_dir)
data['DATE']=0
for i in range(len(data)):
    y=data['YEAR'][i]
    w=data['WEEK'][i]
    d = str(y)+'-W'+str(w)
    r = datetime.datetime.strptime(d + '-6', "%Y-W%W-%w")
    data.loc[i,'DATE']=r

age_groups = data['AGE'].unique()
for id,age_group in enumerate(age_groups):
    data.loc[data['AGE']==age_group, 'AGE'] = desired_age_groups[id]
data = data.drop(columns=['YEAR','WEEK'])
data = data.groupby(by=['DATE','AGE']).sum().squeeze()

# Define a dataframe with the desired format
new_DATE = data.index.get_level_values('DATE').unique() #pd.date_range(start=data.index.get_level_values('DATE').unique()[0],end=data.index.get_level_values('DATE').unique()[-1])
iterables=[new_DATE, data.index.get_level_values('AGE').unique()]
names=['DATE', 'AGE']
index = pd.MultiIndex.from_product(iterables, names=names)
data_new = pd.Series(index=index, name='CASES_100K', dtype=float)

# Merge series with daily date and weekly date
data = data.to_frame()/7
data_new = data_new.to_frame()
merge = data_new.merge(data, how='outer', on=['DATE', 'AGE'])['CASES_100K_y']
merge.name = 'CASES'
# Loop down to series level to perform interpolation of intermittant dates
for age_group in merge.index.get_level_values('AGE').unique():
    interpol = merge.loc[slice(None),age_group].interpolate(method='linear').values
    merge.loc[slice(None),age_group] = interpol

# Per 100K inhabitants
absolute = merge.copy()
for age_group in absolute.index.get_level_values('AGE').unique():    
    absolute.loc[slice(None),age_group] = absolute.loc[slice(None),age_group].values*N.loc[age_group]/100000

#################
## Save result ##
#################

# Write to a new .csv file
merge.to_csv(os.path.join(abs_dir,'data/interim/data_influenza_1718_format_100K.csv'))
absolute.to_csv(os.path.join(abs_dir,'data/interim/data_influenza_1718_format.csv'))
