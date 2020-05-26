#IMPORTANT: the creditcard.csv file has to be in the same folder.

import pandas as pd
import numpy as np
import os

os.system("clear")

shuffled_dataset = pd.read_csv("creditcard.csv")

#######removing V14 outliers#######

#extracting V14 features
v14_data = shuffled_dataset['V14'].loc[shuffled_dataset['Class'] == 1].values

#calculation and printing of IQR and quartiles for V14
q25 = np.percentile(v14_data, 25)
q75 = np.percentile(v14_data, 75)
print('Q25: {}'.format(q25))
print('Q75: {}'.format(q75))
IQR = q75 - q25
print('IQR: {}'.format(IQR))

#Determination of boundaries limits 
v14_cutoff = IQR * 1.6
v14_q25 = q25 - v14_cutoff
v14_q75 = q75 + v14_cutoff
print('Cut Off: {}'.format(v14_cutoff))
print('\n')
print ('V14: ')
print('Lower threshold: {}'.format(v14_q25))
print('upper threshold {}'.format(v14_q75))

outliers = [x for x in v14_data if x < v14_q25 or x > v14_q75]
print('Outliers:{}'.format(outliers))
print('Number of outliers for V14: {}'.format(len(outliers)))


shuffled_dataset = shuffled_dataset.drop(shuffled_dataset[(shuffled_dataset['V14'] > v14_q75) | (shuffled_dataset['V14'] < v14_q25)].index)
print('='*25)



#######removing V12 outliers#######

#extracting V12 features
v12_data = shuffled_dataset['V12'].loc[shuffled_dataset['Class'] == 1].values

#calculating and printing IQR and quartiles for V12
q25 = np.percentile(v12_data, 25)
q75 = np.percentile(v12_data, 75)
IQR = q75 - q25
print('Q25: {}'.format(q25))
print('Q75: {}'.format(q75))
print('IQR: {}'.format(IQR))

#Determination of boundaries limits 
v12_cutoff = IQR * 1.6
v12_q25 = q25 - v12_cutoff
v12_q75 = q75 + v12_cutoff
print('Cut Off: {}'.format(v12_cutoff))
print('\n')
print ('V12: ')
print('Lower threshold: {}'.format(v12_q25))
print('Upper threshold: {}'.format(v12_q75))
outliers = [x for x in v12_data if x < v12_q25 or x > v12_q75]
#for x in v12_data:
    #if x < v12_q25 or x > v12_q75:
     #   outliers.append(x)

print('Outliers: {}'.format(outliers))
print('Number of outliers for V12: {}'.format(len(outliers)))
shuffled_dataset = shuffled_dataset.drop(shuffled_dataset[(shuffled_dataset['V12'] > v12_q75) | (shuffled_dataset['V12'] < v12_q25)].index)
print('='*25)



#######removing V10 outliers#######

#extracting V10 features
v10_data = shuffled_dataset['V10'].loc[shuffled_dataset['Class'] == 1].values

#calculating and printing IQR and quartiles for V10
q25 = np.percentile(v10_data, 25)
q75 = np.percentile(v10_data, 75)
print('Q25: {}'.format(q25))
print('Q75: {}'.format(q75))
IQR = q75 - q25
print('IQR: {}'.format(IQR))

##Determination of boundaries limits 
v10_cutoff = IQR * 1.6
v10_q25 =  q25 - v10_cutoff
v10_q75 = q75 + v10_cutoff
print('Cut Off: {}'.format(v10_cutoff))
print('\n')
print ('V10: ')
print('Lower threshold : {}'.format(v10_q25))
print('Upper threshold : {}'.format(v10_q75))
outliers = [x for x in v10_data if x < v10_q25 or x > v10_q75]
print('Outliers: {}'.format(outliers))
print('Number of outliers for V10: {}'.format(len(outliers)))
shuffled_dataset = shuffled_dataset.drop(shuffled_dataset[(shuffled_dataset['V10'] > v10_q75) | (shuffled_dataset['V10'] < v10_q25)].index)
print('='*25)