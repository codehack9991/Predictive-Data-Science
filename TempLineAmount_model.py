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
