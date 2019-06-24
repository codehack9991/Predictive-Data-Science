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

