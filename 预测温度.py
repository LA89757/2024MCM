import pandas 
import  matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.stattools as st
import statsmodels.api as sm

# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False 

#数据处理
csv_reader=pandas.read_csv(
    'D:\\码仓\\pyhton\\Data\\气温.csv',
    index_col='date'
)

csv_reader.index= pandas.to_datetime(csv_reader.index)#将索引转换为日期格式
#根据每年的气温最高的30天的平均气温得到每年的气温
top_30_days = csv_reader.groupby(csv_reader.index.year)['maximum temperature'].nlargest(30)
average_temperature_per_year = top_30_days.groupby(level=0).mean()
data=pandas.DataFrame(average_temperature_per_year)

print([data.loc[2011:2015]])
#处理缺失值（线性等距插值）
data=data.interpolate(method='linear')
data = data.rename(columns={'maximum temperature': 'tem'})

#ADF检验函数
def tagADF(t):
    result = pandas.DataFrame(index=[
            "Test Statistic Value", "p-value", "Lags Used", 
            "Number of Observations Used", 
            "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"
        ], columns=['tem']
    )
    result['tem']['Test Statistic Value'] = t[0]
    result['tem']['p-value'] = t[1]
    result['tem']['Lags Used'] = t[2]
    result['tem']['Number of Observations Used'] = t[3]
    result['tem']['Critical Value(1%)'] = t[4]['1%']
    result['tem']['Critical Value(5%)'] = t[4]['5%']
    result['tem']['Critical Value(10%)'] = t[4]['10%']
    return result



# 查看趋势图，平稳
data.plot() 
plt.show()
#查看偏自相关图，
plot_acf(data).show()
plt.show()
print('原始序列的ADF检验结果为:\n',tagADF(ADF(data[u'tem'])))  # ADF检验，通过
print(u'差分序列的白噪声检验结果为：\n', acorr_ljungbox(data, lags=1))  # 分别为stat值（统计量）和P值
# P值小于0.05，所以二阶差分后的序列为平稳非白噪声序列

pmax=qmax=int(len(data)/10) 

bic_matrix = [] 
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
#存在部分报错，所以用try来跳过报错。
    try: 
      tmp.append(ARIMA(data, order=(p,2,q)).fit().bic)
    except:
      tmp.append(float('inf'))
  bic_matrix.append(tmp)
#从中可以找出最小值
bic_matrix = pandas.DataFrame(bic_matrix) 
print(bic_matrix)
# 先用stack展平，然后用idxmin找出最小值位置。
stacked_bic = bic_matrix.stack().dropna()
#先用stack展平，然后用idxmin找出最小值位置。
p,q = stacked_bic.idxmin() 
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))
# 取BIC信息量达到最小的模型阶数，结果p为0，q为2，定阶完成。

#拟合模型
model = ARIMA(data, order=(p,0,q)).fit() 
 
#给出一份模型报告
 
print(model.summary())

#作为期10年的预测，返回预测结果、标准误差、置信区间
print(model.forecast(10))



