# 逻辑回归（logistic Regression）

## 基本概念

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

逻辑回归损失函数：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/QQ20170817-172826.png)

当y=1时，假定这个样本为正类。如果此时hθ(x)=1,则单对这个样本而言的cost=0,表示这个样本的预测完全准确。那如果所有样本都预测准确，总的cost=0 
但是如果此时预测的概率hθ(x)=0，那么cost→∞。直观解释的话，由于此时样本为一个正样本，但是预测的结果P(y=1|x;θ)=0, 也就是说预测 y=1的概率为0，那么此时就要对损失函数加一个很大的惩罚项。 
当y=0时，推理过程跟上述完全一致，不再累赘。

将以上两个表达式合并为一个，则单个样本的损失函数可以描述为： 

cost(hθ(x),y)=−yilog(hθ(x))−(1−yi)log(1−hθ(x))

## 代码实例
[点击链接进入逻辑回归代码实战](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/practice/logistic-regression-practice.md)
