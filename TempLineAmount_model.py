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
