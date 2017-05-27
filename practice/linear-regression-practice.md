# linear regression practice

## brief

使用[python数据分析环境搭建](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/practice/python-environment-install.md)中搭建的python环境，一个线性回归的例子，最后用梯度下降算法做最优化。

## 包导入

```
# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

#%config InlineBackend.figure_formats = {'pdf',}
%matplotlib inline  

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')
```
### 样本集导入

```
# 导入当前目录下的样本集文件linear_regression_data1.txt
# 样本feature之间以逗号分隔
data = np.loadtxt('linear_regression_data1.txt', delimiter=',')

# features
X = np.c_[np.ones(data.shape[0]),data[:,0]]

# labels
y = np.c_[data[:,1]]

```

### 画单变量的样本散点图

```
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
```
[更多关于scatter的用法](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/practice/lib-usage/scatter.md)
