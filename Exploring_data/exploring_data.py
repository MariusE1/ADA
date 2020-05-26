import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv',sep=',')

fraud = data[data['Class']==1]
genuine = data[data['Class']==0]

#we look at amounts per transaction for each class

#amount per transaction for fraud transactions
plt.hist(fraud.Amount, color="lightcoral")
plt.title("Amount per transaction for Fraud class")
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.savefig("amounts_fraud.png")
plt.show()
plt.close()

#amount per transaction for genuine transactions
plt.hist(genuine.Amount, color="mediumaquamarine")
plt.title("Amount per transaction for Genuine class")
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.savefig("amounts_genuine.png")
plt.show()
plt.close()


#we look at timing of the transactions for each class
plt.hist(fraud.Time, color="lightcoral")
plt.title("Timing of fraud transactions")
plt.xlabel('Time (in Seconds since first transaction)')
plt.ylabel('Number of Transactions')
plt.savefig("timing_fraud.png")
plt.show()
plt.close()


plt.hist(genuine.Time, color="mediumaquamarine")
plt.title("Timing of genuine transactions")
plt.xlabel('Time (in Seconds since first transaction)')
plt.ylabel('Number of Transactions')
plt.savefig("timing_genuine.png")
plt.show()
plt.close()