import sqlite3
import pandas as pd

df = pd.read_csv('ImageExcel.csv')
df.column = df.columns.str.strip()
connection = sqlite3.connect('DataB.db')
df.to_sql('DataB', connection, if_exists='replace')