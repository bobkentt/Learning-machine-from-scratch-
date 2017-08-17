# 梯度下降法 gradient descent

## 梯度的概念
在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，求得的梯度向量就是(∂f/∂x, ∂f/∂y)T,简称grad f(x,y)或者▽f(x,y)。对于在点(x0,y0)的具体梯度向量就是(∂f/∂x0, ∂f/∂y0)T.或者▽f(x0,y0)，如果是3个参数的向量梯度，就是(∂f/∂x, ∂f/∂y，∂f/∂z)T,以此类推。

那么这个梯度向量求出来有什么意义呢？他的意义从几何意义上讲，就是函数变化增加最快的地方。具体来说，对于函数f(x,y),在点(x0,y0)，沿着梯度向量的方向就是(∂f/∂x0, ∂f/∂y0)T的方向是f(x,y)增加最快的地方。或者说，沿着梯度向量的方向，更加容易找到函数的最大值。反过来说，沿着梯度向量相反的方向，也就是 -(∂f/∂x0, ∂f/∂y0)T的方向，梯度减少最快，也就是更加容易找到函数的最小值。

## 梯度下降的通俗解释
首先来看看梯度下降的一个直观的解释。比如我们在一座大山上的某处位置，由于我们不知道怎么下山，于是决定走一步算一步，也就是在每走到一个位置的时候，求解当前位置的梯度，沿着梯度的负方向，也就是当前最陡峭的位置向下走一步，然后继续求解当前位置梯度，向这一步所在位置沿着最陡峭最易下山的位置走一步。这样一步步的走下去，一直走到觉得我们已经到了山脚。当然这样走下去，有可能我们不能走到山脚，而是到了某一个局部的山峰低处。

从上面的解释可以看出，梯度下降不一定能够找到全局的最优解，有可能是一个局部最优解。当然，如果损失函数是凸函数，梯度下降法得到的解就一定是全局最优解。
[上段话转自](http://www.cnblogs.com/pinard/p/5970503.html)

## 损失函数最优化用到梯度下降算法

* 梯度学习算法做的事情就是最小化损失函数；
* 梯度代表上升最快的方向，负梯度代表下降最快的方向，
* 在深度学习（deep learning）中，也用到了梯度下降算法SGD
* 通过上图的递推公式，不断的迭代参数，找到损失函数最小点
* 学习率是一个超参数（hyper parameter），只有设定好了超参数的值，算法才可以学

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/1.png)

在多元的情况下：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/2.png)

在多元情况下求偏导，其实是在垂直于等高线的方向上，做梯度下降的。

### 学习率（learning rate）
非常重要的概念；
在deep learning 神经网络中， 很多时候算法不收敛的原因就是学习率。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/1.png)

以线性回归模型为例：
  * 如果学习率α过大，会震荡的很厉害，甚至不收敛；
  * 如果过小，可能收敛很慢；
  * 在数学上，是可以找出比较优的学习率，但在工业应用上，找到一个差不多的α就可以了
  * 随着逐步往下走，斜率会逐渐变小
  * 工业上可能拍一组学习率，先去试一下，然后挑出一个
  * 在学术上，有很多算法可以调整学习率，但是都会付出相应的代价

### 公式推导
![](https://github.com/bobkentt/Learning-machine-from-scratch-/raw/master/practice/pic/20170527192321.jpg)

### 代码

```
"""
@parameter
 X     : samples
 y     : labels
 h     : hypothesis
 theta : W
 alpha : learning rate
"""
h = X.dot(theta)
theta = theta - alpha * (1.0/m) * X.T.dot(h-y)
```
