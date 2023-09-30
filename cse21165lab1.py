import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
#Load the data into csv file if multiple sheets are there on the excel file
df = pd.read_excel("C:/Users/DELL/Desktop/ML/lab1.xlsx", sheet_name="Purchase data")
# print(df)

#Dropping/Deleting various columns which are useless
columns_to_drop = ['Candy','Mango','Milk']
df = df.drop(columns = columns_to_drop)
# print(df.head())
df = df.dropna(axis=1)
print(df.head())
# #Making three matrices A , X and C to perform AX=C

# #Making Matrix A
item_quantities = df[['Candies (#)' , 'Mangoes (Kg)' , 'Milk Packets (#)']].values
A=item_quantities
C=df['Payment (Rs)'].values
# print(C)
rank_A=np.linalg.matrix_rank(A)

print("The rank of the matrix A is " , rank_A)

# #---------Inverse of the Matrix A-----------
Inve = np.linalg.pinv(A)
# print("Inverse of the Matrix A is " , Inve)

# #---------Solving For X Matrix to find cost of each product--------
product_costs = np.dot(Inve , C)

print("Product of Costs")
print("Candies Costs:",product_costs[0])
print("Mangoes Costs:",product_costs[1])
print("Milk Packets Costs:",product_costs[2])

# #--------Training the model to add the column of Rich/Poor----------
df['Class'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
# print(df)

# --------------------- IRCTC STOCK PRIZE ------------------------


irctc = pd.read_excel("C:/Users/DELL/Desktop/ML/lab1.xlsx",sheet_name="IRCTC Stock Price")
irctc = irctc.dropna(axis=1)
print(irctc.head())


# Calculating mean and variance of column Price
Mean = st.mean(irctc["Price"])
Var = st.variance(irctc["Price"])
print("Mean of the data: ",Mean)
print("Variance        : ",Var)

#  Select the price data for all Wednesdays and calculate the sample mean. Compare the mean
#  with the population mean and note your observations.
Wed_data = irctc[irctc['Day']=='Wed']
# print(Wed_data.head())
Wed_mean = st.mean(Wed_data['Price'])
# Comparing the sample mean with population mean
if Wed_mean < Mean:
    print("Mean of price data for all Wednesdays is lesser than the mean of all price")
else:
    print("Mean of price data for all Wednesdays is greater than the mean of all price")



# Select the price data for the month of Apr and calculate the sample mean. Compare the
# mean with the population mean and note your observations.
Apr_data = irctc[irctc['Month']=='Apr']
Apr_mean = st.mean(Apr_data['Price'])
print("Mean of price of April month: ",Apr_mean)

if Apr_mean < Mean:
    print("Population mean is greater than the sample mean of April month")
else:
    print("Population mean is greater than the sample mean of April month")
# From the Chg% (available in column I) find the probability of making a loss over the stock.
# (Suggestion: use lambda function to find negative values)

loss = irctc[irctc['Chg%']<0]

loss_pr = len(loss)/len(irctc)
print("Probability of loss over the stock price: ",loss_pr)

'''Calculate the probability of making a profit on Wednesday.'''
profit = irctc[irctc['Chg%']>0]
profit_wed = len(Wed_data)/len(profit) # Wed data we have calculated earlier
print("Probability of profit on Wednesday       : ",profit_wed)
'''Calculate the conditional probability of making profit, given that today is Wednesday.'''
Total_wed = len(Wed_data)

# Total_profit = len(profit)
profit_on_Wed = len(irctc['Chg%']>0)
Cnd_pr = profit_on_Wed/Total_wed
print("Conditional probability: ",Cnd_pr)

'''Make a scatter plot of Chg% data against the day of the week'''
X = irctc['Chg%']
Y = irctc['Day']
plt.scatter(X,Y,label='Data Points',color='blue')
plt.xlabel("Chg%")
plt.ylabel("Week")
plt.show()
