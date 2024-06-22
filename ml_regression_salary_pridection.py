import pandas as pd
salary = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Salary%20Data.csv')

y = salary['Salary']
x = salary[['Experience Years']]
salary.shape

from sklearn.model_selection import train_test_split

x_tarin , x_test , y_train , y_test = train_test_split(x,y,train_size = 0.7 , random_state = 2529)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_tarin,y_train)

model.predict(x_test)

from sklearn.metrics import mean_absolute_percentage_error
error = mean_absolute_percentage_error(y_test,model.predict(x_test))

print(100-(100*error),"% of the accuracy of the salary prediction ml model")
print(x_test)
print("this is the prediction values above table : ",model.predict(x_test))

print("********************************let pridict your salary in the base of data give by*********************************")
val = input("Enter the Experience Years : ")
model.predict([[float(val)]])
