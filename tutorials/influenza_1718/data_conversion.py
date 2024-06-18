############################
## Load required packages ##
############################

import os
import pandas as pd
from datetime import datetime, timedelta

###############
## Load data ##
###############

rel_dir = 'data/raw/ILI_weekly_1718_raw.csv'
data_dir = os.path.join(os.getcwd(),rel_dir)
data = pd.read_csv(data_dir)
# Age groups in dataset
desired_age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)],closed='left')
# Hardcode Belgian demographics (Jan 1, 2018)
N = pd.Series(index=desired_age_groups, data=[620914, 1306826, 7317774, 2130556])

########################
## Perform conversion ##
########################

# convert YEAR-WEEK to the week's midpoint
for i in range(len(data)):
    y=data['YEAR'][i]
    w=data['WEEK'][i]
    d = str(y)+'-W'+str(w)
    r = datetime.strptime(d + '-1', "%Y-W%W-%w") + timedelta(days=4) # Thursday taken as midpoint
    data.loc[i,'DATE']=r
# convert age groups to pd.IntervalIndex
age_groups = data['AGE'].unique()
for id,age_group in enumerate(age_groups):
    data.loc[data['AGE']==age_group, 'AGE'] = desired_age_groups[id]
data = data.drop(columns=['YEAR','WEEK'])
data = data.groupby(by=['DATE','AGE']).sum().squeeze()
# compute weekly total per 100K
absolute = data.copy()
for age_group in absolute.index.get_level_values('AGE').unique():    
    absolute.loc[slice(None),age_group] = absolute.loc[slice(None),age_group].values*N.loc[age_group]/100000

#################
## Save result ##
#################

# Write to a new .csv file
data.to_csv(os.path.join(os.getcwd(),'data/interim/ILI_weekly_100K.csv'))
absolute.to_csv(os.path.join(os.getcwd(),'data/interim/ILI_weekly_ABS.csv'))
