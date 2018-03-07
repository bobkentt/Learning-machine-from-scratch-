# 模型效果优化

根据模型学习曲线可以判断模型的状态。[更多模型学习曲线点击跳转。](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/learning_carve.md)

## 方法
模型不同状态的处理：

过拟合：
* 找更多的数据来学习；
* 增大正则化系数；
* 减少特征个数(不是太推荐)。
*注意：不要以为降维可以解决过拟合问题*

欠拟合：
* 找更多的特征；
* 减小正则化系数；
*除了分析模型状态之外，还可以分析线性模型的权重，以及bad case。*

## 线性模型权重的分析
这里说的线性的模型，主要包括线性或者线性kernel的model，如：
* Linear Regression；
* Logistic Regression；
* Linear SVM。

处理方法是对权重绝对值高或低的特征做：
* 更细化的工作；
* 特征组合。

## 验证集结果中的Bad case的分析
分类问题：
* 哪些训练样本分错了？
* 我们哪部分特征使得它做了这个判定？
* 这些bad case有没有共性？
* 是否有还没挖掘的特性？

回归问题：

哪些样本预测结果差距大，为什么？
