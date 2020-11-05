import pandas as pd
import seaborn as sns

data = pd.read_csv('time_result.csv')
g = sns.catplot(x='model', y='RMSE', kind='box', data=data)

g.savefig('./RMSE.png')

g = sns.catplot(x='model', y='Time', kind='box', data=data)

g.savefig('./Time.png')
