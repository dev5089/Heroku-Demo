

import numpy as np
import pandas as pd

data= pd.read_csv('/content/sample_data/hiring.csv')



data

data['experience']=data['experience'].replace(np.NaN,0)

data

def text_to_int(text):
  text_dict={'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11,0:0}
  return text_dict[text]

data['experience']=data['experience'].apply(lambda x:text_to_int(x))

data

data['test_score(out of 10)']=data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean())

data

x=data.iloc[:,:3]

x


y=data.iloc[:,-1]

y

from sklearn.linear_model import LinearRegression
import pickle

reg=LinearRegression()
reg.fit(x,y)

pickle.dump(reg,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

model

print(model.predict([[2,10.0,10]]))
