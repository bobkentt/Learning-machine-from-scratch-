# 损失函数 loss function

### concept
学习算法的目标是找到最好的参数，然而怎么衡量这个最好的问题？答案是**损失函数（loss function**。

数据标签（label）与预测结果（predict）之间差距叫做损失，衡量损失的函数叫做损失函数（loss function）。

在有监督学习（supervised learning）中，不同模型根据各自不同的目标，都会定义有各自的损失函数（loss function），例如：
* 线性回归（linear regression）中的平方损失函数（loss function）；
* 逻辑回归（logistic regression）中的log损失函数（loss function）；
* svm算法中定义的hinge损失函数（loss function）；

*有很多paper就是对损失函数做优化。*

### eg.
在线性回归中，model定义：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170521165243.png)

在线性回归model中，损失函数是取predict value与label value的方差，定义如下：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170521165348.png)

在参数是一元的情况下：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170521171921.png)

在多元的情况下：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170521171944.png)

平方loss在linear regression的情况下是一个凸函数，这意味着存在全局最优点，并且可能通过某些方法找到全局最优点。在deep learning中没有办法确保找到全局最优点，在工业上找到一个几乎是全局最优点的点，在工业上可以用就行了。

### 线性回归与梯度下降GD
梯度下降算法的细节在：

[点此链接跳转到线性回归与梯度下降GD](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg-base/ch1/gradient_descent.md)

除了梯度下降算法外，还有牛顿法，拟牛顿法等等优化算法。
