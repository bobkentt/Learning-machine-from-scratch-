# 一个CNN调参的例子

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/practice/pic/20170717.png)

> loss function 下降的更线性，应该把learning rate调高，损失函数下降快，反之下降慢；

> loss function 震荡，因为使用SGD，顾应该把data set调大，减少震挡，（data set调大，容易memory out）

> training and valication  accuracy的差距小，所以表现为欠拟合，应该增大model capacity，例如增加neural nodes or hidden neural layer。  
