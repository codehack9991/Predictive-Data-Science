# This script is used to retrain model

import pyodbc
import pandas as pd
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=GCCSWDVDB,2431;'
                      'Database=CCSPROD;'
                      'Trusted_Connection=yes;')

import datetime
 
theday = datetime.date.today()
prevday = theday - datetime.timedelta(days=7)
prevday = prevday.strftime("%Y-%m-%d")
theday = theday.strftime("%Y-%m-%d")
print(prevday)

theday

sql = "select *  from CreditStatsPredictionData where CHDate between '" + prevday + "' and '" + theday + " 23:59:59'" 
credit_x = pd.read_sql(sql,conn)

credit_x = credit_x.fillna(credit_x.median())

credit_x = credit_x.drop(['GFP_ID','ParentDescription'], axis = 1)

credit_x.dtypes

from datetime import datetime
chdate_x = []
for i in range(0, len(credit_x['CHDate'])):
    j = str(credit_x['CHDate'][i].date())
    t = datetime.strptime(j, '%Y-%m-%d').toordinal()
    chdate_x.append(t)
credit_x = credit_x.assign(CHDate = chdate_x)
print (credit_x)

credit_x[300:400]

credit_x = credit_x.drop(['OnExGTotalRiskAmount','OffExGTotalRiskAmount','GTotalTempLine'], axis = 1)

credit_x.dtypes

credit_x = credit_x.drop(['Ctry_Code','GroupNo'], axis = 1)

df_x = credit_x.drop(['TotalTempLineAmount','Line_ID','CountryDesc'], axis = 1)
df_y = credit_x.TotalTempLineAmount
df_y = pd.DataFrame(df_y, columns = ['TotalTempLineAmount'], index = df_x.index.values)

credit_x['CHDate'] = credit_x['CHDate'].astype('float64')
credit_x['TotalTempLineParent'] = credit_x['TotalTempLineParent'].astype('float64')
credit_x.dtypes

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state = 46)
import pickle
predictor = 'temp_amount_prediction.pk'
loaded_model = None
with open('C:/Users/ds46394/Downloads/'+predictor, 'rb') as file:
    loaded_model = pickle.load(file)
loaded_model.partial_fit(x_train, y_train.values.ravel())

# Implementing VAR for x_test
import datetime
theday = datetime.date.today()
prevday = theday - datetime.timedelta(days=100)
prevday = prevday.strftime("%Y-%m-%d")
theday = theday.strftime("%Y-%m-%d")

sql = "select *  from CreditStatsPredictionData where CHDate between '" + prevday + "' and '" + theday + " 23:59:59'" 
credit_x = pd.read_sql(sql,conn)
credit_x = credit_x.fillna(credit_x.median())
df = credit_x
df.dtypes

from datetime import datetime
chdate_x = []
for i in range(0, len(credit_x['CHDate'])):
    j = str(df['CHDate'][i].date())
    t = datetime.strptime(j, '%Y-%m-%d').toordinal()
    chdate_x.append(t)
df = df.assign(CHDate = chdate_x)
data = df
data.index = df.CHDate
data = data.drop(['GFP_ID','ParentDescription','CountryDesc'], axis = 1)
data = data.drop(['OnExGTotalRiskAmount','OffExGTotalRiskAmount','GTotalTempLine'], axis = 1)
data = data.drop(['Ctry_Code','GroupNo'], axis = 1)
data = data.drop(['TotalTempLineAmount','Line_ID'], axis = 1)
# data = data.drop(['OnExTotalRiskAmount','OffExTotalRiskAmount','TotalTempLineParent'], axis = 1)
data.dtypes


from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint_johansen(data,-1,1).eig

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

cols = data.columns
#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,6):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]
       

pred.describe
pred_temp_amount = loaded_model.predict(prediction)
pred_temp_amount

import pickle
filename = 'temp_amount_prediction.pk'

with open('C:/Users/ds46394/Downloads/'+filename, 'wb') as file:
    pickle.dump(loaded_model, file)






