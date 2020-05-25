#IMPORTANT: the creditcard.csv document has to be in the same folder.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
import statistics 

import seaborn as sn
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


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

################################################################################################################################################################
# Random Under sampling (taken from our other script... we should have probably made a function)
################################################################################################################################################################

shuffled_dataset = dataset.sample(frac=1) #samples all rows without replacement, the frac argument specifies the fraction of rows to return in the random sample, with frac=1 the whole dataset is returned in a random order

fraud_data = shuffled_dataset.loc[shuffled_dataset["Class"] == 1] #used for undersampling and oversampling
genuine_data = shuffled_dataset.loc[shuffled_dataset["Class"] == 0] #will be used for oversampling later
genuine_data_cut = shuffled_dataset.loc[shuffled_dataset["Class"] == 0][:492] #used for undersampling

#now we put together in one dataframe both fraud and genuine data
undersampled_dataset = pd.concat([fraud_data,genuine_data_cut])


#we need to shuffle it again else all the fraud rows are from [0:492] and the genuine from [493:984]
shuffled_undersampled_dataset = undersampled_dataset.sample(frac=1)

######################################################################## Spliting Data ##################################################################

#we seperate the target from the values in the datasets
X = dataset.drop("Class", axis=1)
y = dataset["Class"]

X_undersampled = shuffled_undersampled_dataset.drop("Class", axis=1)
y_undersampled = shuffled_undersampled_dataset["Class"]

#the following lists will be used during the K-fold
count = 0
tn_history=[]
fp_history=[]
fn_history=[]
tp_history=[]


#We do a stratified K fold of the original dataset so that we have training and test samples that have an equally weighted amount of fraud class and genuine class.
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #we also perform at each iteration a new split of the dataset we will use for training the Neural Network (basically at each iteration it will be trained with a different 80% of the undersampled dataset. This allows us to do cross validation... Even if we won't test the model on the 20% of test sample coming from the undersampled model but on the test sample coming from the original dataset.
    X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(X_undersampled, y_undersampled, test_size=0.5, shuffle=True)

    X_train_undersampled = X_train_undersampled.values
    X_test_undersampled = X_test_undersampled.values
    y_train_undersampled = y_train_undersampled.values
    y_test_undersampled = y_test_undersampled.values
    
    count += 1
    print("\n =============================== Fold number: ", count," ===============================\n")
    

######################################################################## NN ##################################################################

    #with undersampled dataset as training data
    #architecture [64,32,16]
    
    n_inputs = X_train_undersampled.shape[1]  #there should be 30 dimensions in our problem

    model = keras.Sequential([
        keras.layers.Dense(n_inputs, input_shape=(n_inputs, ), activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation ="sigmoid") #We use a sigmoid function as last-layer activation function since we do binary classification
    ])

    model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.0001), metrics=["accuracy"]) # best loss function for binary classification

    model.fit(X_train_undersampled, y_train_undersampled, validation_split=0.2, batch_size=1, epochs=15, shuffle=True) #we fit the model on a training sample of the undersampled dataset (on 50% of the undersampled to be precise, each training will use a different 50%)

    fraud_predictions = model.predict_classes(X_test, batch_size=50) #we now do predictions for a chosen stratified part of the original dataset (stratified so that there is not only genuine class)

    
    cm = confusion_matrix(y_test, fraud_predictions)
    print("\nConfusion matrix of the fold: ")
    print(cm,"\n")
    
    tn, fp, fn, tp = cm.ravel()
    tn_history.append(tn)
    fp_history.append(fp)
    fn_history.append(fn)
    tp_history.append(tp)
    
    del model #resets the model after each iteration
     
mean_tn = statistics.mean(tn_history)
mean_fp = statistics.mean(fp_history)
mean_fn = statistics.mean(fn_history)
mean_tp = statistics.mean(tp_history)

accuracy = (mean_tp+mean_tn)/(mean_tp+mean_fp+mean_fn+mean_tn)

precision = mean_tp/(mean_tp+mean_fp)

recall = mean_tp/(mean_tp+mean_fn)

F1_score = 2*(recall*precision) / (recall+precision)

print("\nAccuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1_score: ", F1_score,"\n")

#shows the precision recall tradoff

#we plot the last confusion matrix:

ax= plt.subplot()
sn.heatmap(cm, ax = ax, cmap="Greens", fmt="d", annot=False) #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel("Predicted class");ax.set_ylabel("True class"Pos)
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Genuine", "Fraud"]); ax.yaxis.set_ticklabels(["Genuine", "Fraud"])
plt.savefig("Confusion_Matrix_NN.png")
plt.close()



    
    


