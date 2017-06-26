# 神经网络(neural network)

## 神经元模型

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170625005624.jpg)

把许多个这样的神经元按一定层次结构连接起来就得到了神经网络。

以计算机的角度来看，其实就是若干个函数嵌套相连而成。

## 感知机

感知机由两层网络组成，输入层和输出层。如下图所示：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170625010537.jpg)

感知机只有输出层神经元进行激活函数处理，即只有一层功能神经元，只能处理线性可分问题。

线性可分：简单的说就是如果用一个线性函数可以将两类样本完全分开，就称这些样本是“线性可分”的。 更多内容参考：
[](http://blog.csdn.net/u013300875/article/details/44081067)

对于线性不可分的问题，感知机不能够解决：例如下图所示的异或问题：
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170625011545.jpg)

## 多层神经元网络

未处理线性不可分问题，需要使用多层神经元。下图所示，输入层之间和输出层之间包含一层隐层神经元。
隐层神经元和输出层神经元都是含有激活函数的功能神经元。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170625012454.jpg)


更一般的情况是，神经元之间不存在同层相连，也不存在跨层相连，这样的神经网络结构通常称为多层前馈神经网络：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170625012522.jpg)


神经网络的学习过程，就是根据训练数据来调整神经元之间的“连接权”以及每个功能神经元的发指，换言之，神经网络学到的东西，蕴含在连接权与阈值之间。
