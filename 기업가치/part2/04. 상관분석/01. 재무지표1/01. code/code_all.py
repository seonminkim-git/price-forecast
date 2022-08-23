import pandas as pd
import numpy as np
import os
from scipy.cluster import hierarchy as hc
import matplotlib
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from scipy import stats
import math
import sys

warnings.filterwarnings(action='ignore')

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


def cal_corr(df, master, company, y, date, col, flag_sub):
    if flag_sub==True:
        df = df[df[date]<='2019-12-01']
    
    result = df[col].corr().iloc[:,0:1].reset_index()
    result.columns = ['variable', 'corr']
    
    result = pd.merge(result, master, on='variable', how='left')
    result = result.drop(0, axis=0).reset_index(drop=True)
    result['company'] = company
    result['corr'] = result['corr'].round(3)
    
    result = result[['company', 'variable', 'corr', 'group','importance']]
    return result



def scatter_plot(df, result, flag_sub, y_name):
    if flag_sub==True:
        df = df[df[date]<='2019-12-01']
    
    for k in result['group'].unique():
        print('\n')
        print(k)
        target_var = result[result['group']==k]['variable']

        i,j = 0,0
        PLOTS_PER_ROW = 3
        num_rows = math.ceil(len(target_var)/PLOTS_PER_ROW)
        if num_rows==1: num_rows=2 

        fig, axs = plt.subplots(num_rows, PLOTS_PER_ROW, figsize=(20, 5*num_rows))
        for var in target_var:
            sns.regplot(data=df, x=var, y=y_name, ax= axs[i][j])
            axs[i][j].set_title(var+'\nCorrelation :'+'%.3f' %result[result['variable']==var]['corr'])
            j+=1
            if j%PLOTS_PER_ROW==0:
                i+=1
                j=0
        fig.tight_layout()
#         fig.subplots_adjust(top=0.93)
#         fig.suptitle(k, size=16)
        plt.show()
    
    
def scatter_plot_for_search(df, result, flag_sub, y_name, company):
    if flag_sub==True:
        df = df[df[date]<='2019-12-01']
    
    for k in result['group'].unique():
        print('\n')
        print(k)
        target_var = result[result['group']==k]['variable']
    
        plt.figure(figsize = (6, 5))
        for var in target_var:
            print(company)
            print(var)
            sns.regplot(data=df, x=var, y=y_name)
            plt.show()