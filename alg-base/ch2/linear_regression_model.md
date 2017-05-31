# 线性回归（linear_regression_model）

## 模型描述
线性模型通过线性组合来预测函数的值。

![](http://images.cnitblog.com/blog2015/633472/201503/262037556613399.jpg)

现在我们需要根据给定的X求解W的值,假设有训练数据：

![](http://images.cnitblog.com/blog2015/633472/201503/262041198028564.jpg)

写成：

![](http://images.cnitblog.com/blog2015/633472/201503/262042295678545.jpg)


## 求解方法

### 最小二乘法
基于均方误差最小化来进行模型求解的方法叫“最小二乘法”，在线性回归中，最小二乘法就是试图找到一条直线，使所有样本到直线上的欧式距离之和最小。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/20170511184436.png)

是一个凸函数，想下y=x^2是凸函数。对其关于w和x求偏导，得到：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/20170511184601.png)

令等式等于0，求得：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/20170511185148.png)

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/20170511185242.png)

更一般的情况：损失函数：

![](http://images.cnitblog.com/blog2015/633472/201503/262045294426265.jpg)

在矩阵满秩的时候，有解：
![](http://images.cnitblog.com/blog2015/633472/201503/262057197703308.jpg)


然并卵，在实际中，往往是矩阵不满秩（条件向量多于数据条数，如金融数据），矩阵求逆矩阵的计算量也是很大的，所以工业上常用一些最优化算法如梯度下降算法。

### 梯度下降法
假设给定数据集：

![](http://images.cnitblog.com/blog2015/633472/201503/262115482708161.jpg)

对损失函数求偏导如下：

![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/WindowsLiveWriter/1_1270E/image_thumb_12.png)


下面是更新的过程，也就是θi会向着梯度最小的方向进行减少。θi表示更新之前的值，-后面的部分表示按梯度方向减少的量，α表示步长，也就是每次按照梯度减少的方向变化多少。

![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/WindowsLiveWriter/1_1270E/image_thumb_14.png)
