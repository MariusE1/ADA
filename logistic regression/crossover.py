#Splitting the data in between test and training

from sklearn.model_selection import train_test_split
from under_oversampling import fraud_data, genuine_data_cut
# Whole dataset

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0) # we take a test size of 25 percent here

print("Dataset contains :", len(x_train), "number of train transactions")
print("Dataset contains :", len(x_test), "number of test transactions")
print("Dataset contains :", len(x_train)+len(x_test), "number of total transactions")


# We chose to undersample the dataset of genuine dataset in order to get the same sample size

x_train_under, x_test_under, y_train_under, y_test_under = train_test_split(x_under,
                                                                            y_under,
                                                                            test_size = 0.25,
                                                                            random_state = 0)


print("#####################################################################")
print("Dataset contains :", len(x_train_under), "number of train transactions // undersampled")
print("Dataset contains :", len(x_test_under), "number of test transactions // undersampled")
print("Dataset contains :", len(x_train_under)+len(x_test_under), "number of total transactions // undersampled")

