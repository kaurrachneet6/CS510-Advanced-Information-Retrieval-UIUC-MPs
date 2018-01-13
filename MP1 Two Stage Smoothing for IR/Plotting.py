# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 01:50:18 2017

@author: Rachneet Kaur
"""

import pandas as pd
import matplotlib.pyplot as plt

file1='CSVs/dirichlet-prior-sensitivity-ap88-89.csv'
file2='CSVs/dirichlet-prior-sensitivity-ziff1-2.csv'
file3= 'CSVs/jelinek-mercer-sensitivity-ap88-89.csv'
file4='CSVs/jelinek-mercer-sensitivity-ziff1-2.csv'
file5='CSVs/two-stage-sensitivity-ap88-89.csv'
file6='CSVs/two-stage-sensitivity-ziff1-2.csv'
file7='CSVs/two-stage-sensitivity-ap88-89LongVerbose.csv'
file8='CSVs/two-stage-sensitivity-ziff1-2LongVerbose.csv'

data1=pd.read_csv(file5)
data2=pd.read_csv(file7)
data3=pd.concat([data1,data2],ignore_index='False')
data3.to_csv('CSVs/two-stage-sensitivity-ap88-89All.csv',index=False)
file9='CSVs/two-stage-sensitivity-ap88-89All.csv'

data4=pd.read_csv(file6)
data5=pd.read_csv(file8)
data6=pd.concat([data4,data5],ignore_index='False')
data6.to_csv('CSVs/two-stage-sensitivity-ziff1-2All.csv',index=False)
file10='CSVs/two-stage-sensitivity-ziff1-2All.csv'

files=[file1, file2, file3, file4, file9, file10]
params=['Mu', 'Mu','Lambda','Lambda','Mu, Lambda = 0.7','Mu, Lambda = 0.7']

def plot(i,f):
        data=pd.read_csv(f)
        short_keyword=data[data['queryset']=='short-keyword']
        long_keyword=data[data['queryset']=='long-keyword']
        short_verbose=data[data['queryset']=='short-verbose']
        long_verbose=data[data['queryset']=='long-verbose']
        plt.plot(short_keyword['parameter'],short_keyword['map'],'*-',label='Short Keyword')
        plt.plot(long_keyword['parameter'],long_keyword['map'],'s-',label='Long Keyword')
        plt.plot(short_verbose['parameter'],short_verbose['map'],'d-',label='Short Verbose')
        plt.plot(long_verbose['parameter'],long_verbose['map'],'o-',label='Long Verbose')
        plt.xlabel('Parameter - '+ params[i])
        plt.ylabel('Mean Average Precision')
        plt.title('Sensitivity of precision for '+ f[5:-4])
        plt.ylim([0,0.5])
        plt.legend()
        plt.savefig(f[5:-4]+'.svg',format='svg')
        plt.show()
        
plt.close()
for i,f in enumerate(files):
    plot(i,f)
