# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:36:43 2021

@author: Jiahao Chen
"""
import pandas as pd
import seaborn as sns

data = pd.read_csv(r'NTN-data.csv')
data.head()

data_var = data[['Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4', 'pH']]
data_var
sns.heatmap(data_var.corr(),vmin=-1, vmax=1, center=0, cbar=True)
