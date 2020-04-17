import pandas as pd

data = pd.read_csv('crime.csv', index_col=0)
print(data)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols])
y = np.array(data[target])
#X, y = shuffle(X, y, random_state=1)

print(X[:,0])
plt.plot(X[:,0], y, 'o')
#plt.plot(data["Education"], data["Crime"])
plt.title("Crime and Education") 
plt.xlabel("Education") 
plt.ylabel("Crime") 
plt.show()


print(X[:,1])
plt.plot(X[:,1], y, 'o')
#plt.plot(data["Education"], data["Crime"])
plt.title("Crime and Poilce") 
plt.xlabel("Poilce") 
plt.ylabel("Crime") 
plt.show()

print(X[:,2])
plt.plot(X[:,2], y, 'o')
#plt.plot(data["Education"], data["Crime"])
plt.title("Crime and Income") 
plt.xlabel("Income") 
plt.ylabel("Crime") 
plt.show()

print(X[:,3])
plt.plot(X[:,3], y, 'o')
#plt.plot(data["Education"], data["Crime"])
plt.title("Crime and Inequality") 
plt.xlabel("Inequality") 
plt.ylabel("Crime") 
plt.show()

education = (X[:,0])
from sklearn.model_selection import train_test_split
EducationTrain, EducationTest, yTrain, yTest = train_test_split(education, y, test_size = 0.5, random_state = 0)
print("\n")
print("Education Train and Test")
print("Train Half")
print(EducationTrain)
print("Test Half")
print(EducationTest)

poilce = (X[:,1])
from sklearn.model_selection import train_test_split
PoilceTrain, PoilceTest, yTrain2, yTest2 = train_test_split(poilce, y, test_size = 0.5, random_state = 0)
print("\n")
print("Poilce Train and Test")
print("Train Half")
print(PoilceTrain)
print("Test Half")
print(PoilceTest)

income = (X[:,2])
from sklearn.model_selection import train_test_split
IncomeTrain, IncomeTest, yTrain, yTest = train_test_split(income, y, test_size = 0.5, random_state = 0)
print("\n")
print("Income Train and Test")
print("Train Half")
print(IncomeTrain)
print("Test Half")
print(IncomeTest)

inequality = (X[:,3])
from sklearn.model_selection import train_test_split
InequalityTrain, InequalityTest, yTrain, yTest = train_test_split(inequality, y, test_size = 0.5, random_state = 0)
print("\n")
print("Income Train and Test")
print("Train Half")
print(InequalityTrain)
print("Test Half")
print(InequalityTest)



lr = LinearRegression()
print("\n")
print(" 1. Inequality Train Linear Regression")

InequalityTrain = data.iloc[:,3].values.reshape(-1, 1)
y = data.iloc[:,1].values.reshape(-1, 1)
lr.fit(InequalityTrain, y)
yTest_pred = lr.predict(InequalityTrain)
plt.scatter(InequalityTrain, y)
plt.plot(InequalityTrain, yTest_pred, color='orange')
plt.figure(4)
plt.show
print("\n")

print("2 .Poilce Train Linear Regression")
PoilceTrain = data.iloc[:,1].values.reshape(-1, 1)
yTest = data.iloc[:,0].values.reshape(-1, 1)
lr.fit(PoilceTrain, yTest)
yTest_pred = lr.predict(PoilceTrain)
plt.scatter(PoilceTrain, yTest)
plt.plot(PoilceTrain, yTest_pred, color='green')
plt.figure(2)
plt.show
print("\n")

print("3 .Income Train Linear Regression")
IncomeTrain = data.iloc[:,2].values.reshape(-1, 1)
yTest = data.iloc[:,0].values.reshape(-1, 1)
lr.fit(IncomeTrain, yTest)
yTest_pred = lr.predict(IncomeTrain)
plt.scatter(IncomeTrain, yTest)
plt.plot(IncomeTrain, yTest_pred, color='purple')
plt.figure(3)
plt.show
print("\n")

print("4. Education Train Linear Regression")
EducationTrain = data.iloc[:,0].values.reshape(-1, 1)
yTest = data.iloc[:,1].values.reshape(-1, 1)
lr.fit(EducationTrain, yTest)
yTest_pred = lr.predict(EducationTrain)
plt.scatter(EducationTrain, yTest)
plt.plot(EducationTrain, yTest_pred, color='red')
plt.figure(1)
plt.show



