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
# np.ones(data.shape[0])是偏执项b
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

![](./pic/20170527145408.jpg)

### loss function

线性回归损失函数：

![](http://images.cnitblog.com/blog2015/633472/201503/262045294426265.jpg)

代码如下：

```
# 计算损失函数
def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0

    h = X.dot(theta)

    J = 1.0/(2*m)*(np.sum(np.square(h-y)))

    return J
```

#### 关于numpy运算:
* *nmpy.sum：
[点我跳转查看官方文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html#numpy.sum)*
* *nmpy.square：
[点我跳转查看官方文档](https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html#numpy.square)*
* *更多运算请参考：
[点我跳转查看官方文档](https://docs.scipy.org/doc/numpy/reference/routines.math.html)*

### 梯度下降算法

#### 公式：

根据梯度下降算法的推导公式：

![](./pic/20170527192321.jpg)

```
def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=2000):
    m = y.size
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1.0/m) * X.T.dot(h-y)
        J_history[iter] = computeCost(X,y,theta)
    return (theta, J_history)
```
