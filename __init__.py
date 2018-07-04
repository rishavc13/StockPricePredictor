#!/usr/bin/python3

from flask import Flask,render_template,request,redirect,url_for
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import math

# Creating Application Name
app=Flask(__name__)

# Defining Routes
@app.route('/result',methods=['POST','GET'])
def result():
	if request.method=='POST':
		company=request.form['menu']
		day=request.form['day']
		month=request.form['month']
		year=request.form['year']
		cal=(str(year)+'-'+str(month)+'-'+str(day))
		print('Date queried :',cal)
		print('Company queried :',company)
		amount,error=function(company,cal)
		print('Predicted Amount:',amount)
		return render_template('result.html',output=amount,error=error)

def function(company,cal):
	# Importing dataset in csv format
	df=pd.read_csv('static/Apple.csv')
	
	# Checking for null elements if any
	df.isnull().any()
	
	original=df.Close[df['Date']==cal]

	# Selecting features
	features=['Open','High','Low','Adj Close','Volume']
	target=['Close']
	date=df.Date

	# Finding size of dataset
	size=df.shape[0]

	# Setting the training size
	train_size=0.8 * size

	# Splitting Dataset
	x_train=df[features].loc[:train_size]
	x_target=df[target].loc[:train_size]
	y_train=df[features].loc[train_size:]
	y_target=df[target].loc[train_size:]

	# Calling the LinearRegression function
	lg=LinearRegression()
	trained=lg.fit(x_train,x_target)

	# Making prediction
	prediction=trained.predict(y_train)

	# Joining Predicted output and Date columns for further analysis
	cols={'Date':df['Date'].loc[math.ceil(train_size):],'Prediction':prediction.ravel()}
	pred_df=pd.DataFrame(cols,index=np.arange(start=math.ceil(train_size),stop=size,step=1))
	#print(pred_df)
	joined=y_train.join(pred_df,how='inner')
	#print(joined)

	x=joined['Date']
	y=joined['Prediction']
	x1=list(x)
	y1=list(y)
	# Displaying predicted graph
	plt.figure(figsize=(30,15))
	plt.plot(x,y,color='red',label='Predicted')
	plt.xticks(rotation=90)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.plot(x,y_target,color='green',label='Actual')
	plt.legend()
	plt.savefig('static/abc.jpg')
	plt.close()

	# Finding error
	mean_error=mean_absolute_error(y_target,prediction)
	error=original-joined.Prediction[joined['Date']==cal]
	print('Mean Absolute Error:',error)
	if cal in x1:
		return(round(y1[x1.index(cal)],2),round(mean_error,2))
	else:
		return('not found','not found')

	

@app.route('/')
def homepage():
	return render_template('main.html')

if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0',port=80)

