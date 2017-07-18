# 模型效果优化

根据模型学习曲线可以判断模型的状态。

[学习曲线](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/learning_carve.md)


## 方法

```
不同模型状态处理
 过拟合
找更多的数据来学习
增大正则化系数
减少特征个数(不是太推荐)
注意：不要以为降维可以解决过拟合问题

 欠拟合
找更多的特征
减小正则化系数
```


除了模型状态之外，还可以对线性模型的权重进行分析，以及一些bad case的分析。

## 线性模型的权重分析

```
线性模型的权重分析
 过线性或者线性kernel的model
Linear Regression
Logistic Regression
LinearSVM
…
```

对权重绝对值高/低的特征
> 做更细化的工作

> 特征组合

## Bad-case分析
分类问题

> 哪些训练样本分错了？

> 我们哪部分特征使得它做了这个判定？

> 这些bad cases有没有共性

> 是否有还没挖掘的特性

回归问题
> 哪些样本预测结果差距大，为什么？

>  …
