# logistic Regression 模型

## concept

linear regression在解决分类问题的时候健壮性很差，现实中很多时候样本集中包含脏数据，例如电商数据中的刷单行为等。当噪声来的时候，模型很容易走偏。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704311.jpg)

在分类问题中希望有一个函数能输出分类，所以想到了在线性回归模型外加入一个压缩函数，logistics就是加入了sigmoid函数的广义线性模型。

## sigmoid压缩函数，
* y =1  /  (  1 + e^(-z) )
* 逻辑回归是对概率p做拟合
* 在神经网络中有重要的应用
* 是0和1之间的一个值，不包括0和1
* 当x取0的时候，y取0.5

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704312.jpg)

## decision boundary

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704313.png)

## 损失函数
逻辑回归L2损失不是凸函数。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704314.jpg)

## 多分类问题
