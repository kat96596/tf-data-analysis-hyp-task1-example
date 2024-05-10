import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, probplot, norm, ttest_ind, cramervonmises_2samp
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.sandbox.stats.multicomp import multipletests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Так как в конечном итоге главной целью является увеличение прибыли, в качестве целевой метрики можно выбрать NPV групп. 
#Тогда основная и альтернативные гипотезы будут такими:
#H0: Среднее значение NPV в тестовой и контрольной группах не различается
#H1:Среднее значение NPV в тестовой группе (m0) больше, чем среднее значение NPV в контрольной группе(m) (m < m0)

#Контрольная метрика
#Во время тестирования важно следить за тем, чтобы в тестовой группе не упала конверсия, поэтому примем за контрольную метрику конверсию из заявки в продажу.
#Прокси метрика
#кумулятивное NPV
#кумулятивное среднее NPV
#Эти метрики сонаправлены с целевой и созревают быстрее – их можно отслеживать в динамике

data = pd.read_csv('hist_telesales.csv')
data.head()
data['NPV'].hist(bins = 200)
alpha = 0.05
beta = 0.2
variance = data['NPV'].var()
d = data['NPV'].mean() * alpha
size = (2 * variance * (norm.ppf(1.0 - alpha) - norm.ppf(beta)) ** 2) / (d ** 2)
round(size)

sample_id = 18294 #номер выборки
chat_id = 790527898 # Ваш chat ID, не меняйте название переменной

control = pd.read_csv('Контроль.csv')
test = pd.read_csv('Тест.csv')
control.head()

plt.figure(figsize=(12, 7))

plt.plot(test['ID'], test['NPV'].cumsum(), label='Тестовая группа')
plt.plot(control['ID'], control['NPV'].cumsum(), label='Контрольная группа')
plt.title('Кумулятивное NPV тестовой и контрольной группы')
plt.ylabel('Кумулятивное NPV')
plt.xlabel('Заявки')
plt.legend()
plt.show()

control['cum_sale_flag'] = control['Флаг продажи'].cumsum()
test['cum_sale_flag'] = test['Флаг продажи'].cumsum()

plt.figure(figsize=(12, 7))

plt.plot(test['ID'], test['cum_sale_flag'] / len(control), label='Тестовая группа')
plt.plot(control['ID'], control['cum_sale_flag'] / len(control), label='Контрольная группа')
plt.title('Кумулятивная конверсия тестовой и контрольной группы')
plt.ylabel('Кумулятивная конверсия')
plt.xlabel('Заявки')
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 9))

ax1 = plt.subplot(2, 2, 1)
ax1.hist(control['NPV'], bins=300)
plt.xlim(-500, 2000)
plt.title('Контрольная группа')
plt.xlabel('NPV')

ax2 = plt.subplot(2, 2, 2, sharey = ax1)
ax2.hist(test['NPV'], bins=300)
plt.xlim(-500, 2000)
plt.xlabel('NPV')

fig.suptitle('Распределение NPV')
plt.title('Тестовая группа')
plt.show()

stat, p_value = ttest_ind(test['NPV'], control['NPV'], alternative='greater')

percent = round(test['NPV'].mean()/control['NPV'].mean()-1, 3)

print(f'Процентное различие между средними равно {percent}')
print(f'p-value: {p_value}')

if p_value < 0.05:
    print('Отвергаем нулевую гипотезу: между выборками есть значимая разница')
else:
    print('Не отвергаем нулевую гипотезу, средние можно считать равными')
  
# def solution(x_success: int, 
#             x_cnt: int, 
 #            y_success: int, 
 #            y_cnt: int) -> bool:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
   # return ... # Ваш ответ, True или False
