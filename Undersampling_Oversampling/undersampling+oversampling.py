#WARNING: IN ORDER TO USE THIS SCRIPT, YOU NEED TO HAVE THE ENTIRE CREDITCARD.CSV DATASET IN THE SAME FOLDER

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn #allows us to do nice correlation matrices.

from sklearn.preprocessing import StandardScaler


os.system("clear")

dataset = pd.read_csv("creditcard.csv")
scaler = StandardScaler()

#this adds two rows at the end of the dataset containing the scaled values of the amount and time column 
dataset["scaled_amount"] = scaler.fit_transform(dataset["Amount"].values.reshape(-1,1))
dataset["scaled_time"] = scaler.fit_transform(dataset["Time"].values.reshape(-1,1))

#we can now remove the old ones which are not scaled
dataset.drop(["Time"], axis=1, inplace=True)
dataset.drop(["Amount"], axis=1, inplace=True)

#now we want to put the new columns (scaled columns) back to their initial positions (where the Time and Amount columns were)
scaled_amount = dataset["scaled_amount"]
scaled_time = dataset["scaled_time"]

dataset.drop(["scaled_amount","scaled_time"], axis=1, inplace=True)
dataset.insert(0,"scaled_amount",scaled_amount)
dataset.insert(1,"scaled_time",scaled_time)

###############################################################################################################################################################################################################
# Random Under sampling
###############################################################################################################################################################################################################


#We know that there is 492 fraud cases in the dataset from the description. We need to take all the fraud rows and take 492 random genuine rows.
#Check the count of the 2 classes
print(dataset["Class"].value_counts())


#First we can shuffle the whole data set with pandas
shuffled_dataset = dataset.sample(frac=1) #samples all rows without replacement, the frac argument specifies the fraction of rows to return in the random sample, with frac=1 the whole dataset is returned in a random order

fraud_data = shuffled_dataset.loc[shuffled_dataset["Class"] == 1] #used for undersampling and oversampling
genuine_data = shuffled_dataset.loc[shuffled_dataset["Class"] == 0] #will be used for oversampling later
genuine_data_cut = shuffled_dataset.loc[shuffled_dataset["Class"] == 0][:492] #used for undersampling

#now we put together in one dataframe both fraud and genuine data
undersampled_dataset = pd.concat([fraud_data,genuine_data_cut])
print("Not shuffled undersampled dataset")
print(undersampled_dataset)

#we need to shuffle it again else all the fraud rows are from [0:492] and the genuine from [493:984]
shuffled_undersampled_dataset = undersampled_dataset.sample(frac=1)
print("Shuffled undersampled dataset")
print(shuffled_undersampled_dataset)




###############################################################################################################################################################################################################
# Correlation matrices
###############################################################################################################################################################################################################




################## with Seaborn ############################

#figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,11))

# Entire dataset
f = plt.figure(figsize=(15,11))
corr = dataset.corr()
sn.heatmap(corr, cmap="RdBu_r", annot_kws={"size":20})
plt.title("Entire/Original dataset correlation matrix", fontsize=16);
plt.savefig("Original_seaborn_correlation_matrix.png")
plt.show()

#Undersampled dataset
f = plt.figure(figsize=(15,11))
undersampled_corr = shuffled_undersampled_dataset.corr()
sn.heatmap(undersampled_corr, cmap="RdBu_r", annot_kws={"size":20})
plt.title("Undersampled dataset correlation matrix", fontsize=16);
plt.savefig("Undersampled_seaborn_correlation_matrix.png")
plt.show()


################## with matplotlib ############################

# Entire dataset
f = plt.figure(figsize=(15, 11))
plt.matshow(dataset.corr(), fignum=f.number)
plt.xticks(range(dataset.shape[1]), dataset.columns, fontsize=14, rotation=45)
plt.yticks(range(dataset.shape[1]), dataset.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Entire/Original dataset correlation matrix", fontsize=16);
plt.savefig("Original_matplotlib_correlation_matrix.png")
plt.show()

#Undersampled dataset
f = plt.figure(figsize=(15, 11))
plt.matshow(shuffled_undersampled_dataset.corr(), fignum=f.number)
plt.xticks(range(shuffled_undersampled_dataset.shape[1]), shuffled_undersampled_dataset.columns, fontsize=14, rotation=45)
plt.yticks(range(shuffled_undersampled_dataset.shape[1]), shuffled_undersampled_dataset.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Undersampled dataset correlation matrix", fontsize=16);
plt.savefig("Undersampled_matplotlib_correlation_matrix.png")
plt.show()

###############################################################################################################################################################################################################
# Random Over sampling
###############################################################################################################################################################################################################

print("\nthe computation of the oversampled dataset takes some time...\n")

#we need to recall that we already have arrays containing all fraud data and all genuine data. For our oversampled dataset, we will simply duplicate the fraud data 500 times and shuffle the dataset again.

count = 0
while count<500 :
    if count == 0:  
        oversampled_dataset = pd.concat([fraud_data,genuine_data])
        count += 1
    else:
        oversampled_dataset = pd.concat([oversampled_dataset,fraud_data])
        count += 1
        
print(oversampled_dataset["Class"].value_counts()) #this allows us to see the new balance between both classes.
shuffled_oversampled_dataset = oversampled_dataset.sample(frac=1) #shuffles the new dataset again

f = plt.figure(figsize=(15,11))
oversampled_corr = shuffled_oversampled_dataset.corr()
cmap = sn.diverging_palette(150, 4, s=80, n=100)
sn.heatmap(oversampled_corr, cmap=cmap, annot_kws={"size":20})
plt.title("Oversampled dataset correlation matrix", fontsize=16);
plt.savefig("Oversampled_seaborn_correlation_matrix.png")
plt.show()
