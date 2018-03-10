# 数据集分类
在有监督(supervise)的机器学习中，数据集常被分成2~3个，即：训练集(train set) 验证集(validation set) 测试集(test set)。
训练集用来估计模型，验证集用来确定网络结构或者控制模型复杂程度的参数，而测试集则检验最终选择最优的模型的性能如何。
Ripley, B.D（1996）在他的经典专著Pattern Recognition and Neural Networks中给出了这三个词的定义。

```
Training set:
 A set of examples used for learning, which is to fit the parameters [i.e., weights] of the classifier.
Validation set:
A set of examples used to tune the parameters [i.e., architecture, not weights] of a classifier, for example to choose the number of hidden units in a neural network.
Test set:
 A set of examples used only to assess the performance [generalization] of a fully specified classi
 ```
