import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('taxi.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text="Number of Weekly Rides Should be {}".format(math.floor(output)))

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080)

    app.run(debug=True)

#  Machine learning code

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# data = pd.read_csv('taxi.csv')
# # print(data.head())

# data_x = data.iloc[:,0:-1].values
# data_y = data.iloc[:,-1].values
# print(data_y)

# X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

# reg = LinearRegression()
# reg.fit(X_train,y_train)

# print("Train Score:", reg.score(X_train,y_train))
# print("Test Score:", reg.score(X_test,y_test))

# pickle.dump(reg, open('taxi.pkl','wb'))

# model = pickle.load(open('taxi.pkl','rb'))
# print(model.predict([[80, 1770000, 6000, 85]]))

#  Data set code

# Priceperweek,Population,Monthlyincome,Averageparkingpermonth,Numberofweeklyriders
# 15,1800000,5800,50,192000
# 15,1790000,6200,50,190400
# 15,1780000,6400,60,191200
# 25,1778000,6500,60,177600
# 25,1750000,6550,60,176800
# 25,1740000,6580,70,178400
# 25,1725000,8200,75,180800
# 30,1725000,8600,75,175200
# 30,1720000,8800,75,174400
# 30,1705000,9200,80,173920
# 30,1710000,9630,80,172800
# 40,1700000,10570,80,163200
# 40,1695000,11330,85,161600
# 40,1695000,11600,100,161600
# 40,1690000,11800,105,160800
# 40,1630000,11830,105,159200
# 65,1640000,12650,105,148800
# 102,1635000,13000,110,115696
# 75,1630000,13224,125,147200
# 75,1620000,13766,130,150400
# 75,1615000,14010,150,152000
# 80,1605000,14468,155,136000
# 86,1590000,15000,165,126240
# 98,1595000,15200,175,123888
# 87,1590000,15600,175,126080
# 77,1600000,16000,190,151680
# 63,1610000,16200,200,152800
