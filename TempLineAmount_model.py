import pyodbc
import pandas as pd
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=GCCSWDVDB,2431;'
                      'Database=CCSPROD;'
                      'Trusted_Connection=yes;')
# excelfile = pd.ExcelFile(r'C:\Users\ds46394\Downloads\CreditStats_data.xlsx')
# cursor = conn.cursor()
# cursor.execute('select *  from CreditStatsPredictionData')
# credit_x = excelfile.parse('x_train1')

sql = "select *  from CreditStatsPredictionData"
credit_x = pd.read_sql(sql,conn)
credit_x = credit_x.fillna(credit_x.median())
credit_x.shape
credit_x = credit_x.drop(['GFP_ID','ParentDescription'], axis = 1)

from datetime import datetime
chdate_x = []
for i in range(0, len(credit_x['CHDate'])):
    j = str(credit_x['CHDate'][i].date())
    t = datetime.strptime(j, '%Y-%m-%d').toordinal()
    chdate_x.append(t)
credit_x = credit_x.assign(CHDate = chdate_x)
print (credit_x)
credit_x[300:400]
credit_x.corr()

import numpy as np
corr = np.corrcoef(credit_x[0:10000], rowvar = 0)
w,v = np.linalg.eig(corr)
print (w)

credit_x = credit_x.drop(['Ctry_Code','GroupNo'], axis = 1)
credit_x.dtypes
credit_x = credit_x.drop(['OnExGTotalRiskAmount','OffExGTotalRiskAmount','GTotalTempLine'], axis = 1)
credit_x['CHDate'] = credit_x['CHDate'].astype('float64')
credit_x['TotalTempLineParent'] = credit_x['TotalTempLineParent'].astype('float64')
df_x = credit_x.drop(['TotalTempLineAmount'], axis = 1)
df_y = credit_x.TotalTempLineAmount
df_y = pd.DataFrame(df_y, columns = ['TotalTempLineAmount'], index = df_x.index.values)

from sklearn.model_selection import train_test_split
credit_x.dtypes
df_x = credit_x.drop(['TotalTempLineAmount','Line_ID','CountryDesc'], axis = 1)
df_x.dtypes
df_y = credit_x.TotalTempLineAmount
df_y = pd.DataFrame(df_y, columns = ['TotalTempLineAmount'], index = df_x.index.values)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state = 9)
print(x_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train_scaled = pd.DataFrame(scaler.transform(x_train), index = x_train.index.values, columns = x_train.columns.values)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), index = x_test.index.values, columns = x_test.columns.values)
from sklearn.svm import SVR
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=5, epsilon=.1,
               coef0=1)
pred  = svr_poly.fit(x_train, y_train.values.ravel()).predict(x_test)
from sklearn.metrics import r2_score
test_score = r2_score(y_test, pred)
test_score

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(oob_score = True, random_state = 46, n_jobs = -1, n_estimators = 100)
rfr.fit(x_train, y_train.values.ravel())

from sklearn.metrics import r2_score
pred_temp_amount = rfr.predict(x_test)
test_score = r2_score(y_test, pred_temp_amount)
test_score

from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=5, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
nn = nn.fit(x_train, y_train.values.ravel())
from sklearn.metrics import r2_score
pred_temp_amount = nn.predict(x_test)
test_score = r2_score(y_test, pred_temp_amount)
test_score

print('Test data R-2 score: %.3f' % (test_score))
print (y_test[0:12])

import pickle
filename = 'temp_amount_prediction.pk'

with open('C:/Users/ds46394/Downloads/'+filename, 'wb') as file:
    pickle.dump(nn, file)
predictor = 'temp_amount_prediction.pk'
loaded_model = None
with open('C:/Users/ds46394/Downloads/'+predictor, 'rb') as file:
    loaded_model = pickle.load(file)
predictions = loaded_model.predict(x_test)
# for i in range(0, len(predictions)):
#     print predictions['i']
