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

## 数据集划分

1. 留出法（hold-out）：直接从数据集中划分出训练集和测试集（通常50%到66%用于训练集）；
2. 交叉验证法（cross-validation）：将原始数据分成K组(一般是均分),将每个子集数据分别做一次验证集,其余的K-1组子集数据作为训练集,这样会得到K个模型,用这K个模型最终的验证集的分类准确率的平均数作为此K-CV下分类器的性能指标.K一般大于等于2,实际操作时一般从3开始取,只有在原始数据集合数据量小的时候才会尝试取2.K-CV可以有效的避免过学习以及欠学习状态的发生,最后得到的结果也比较具有说服性。
3. 如果设原始数据有N个样本,那么LOO-CV就是N-CV,即每个样本单独作为验证集,其余的N-1个样本作为训练集,所以LOO-CV会得到N个模型,用这N个模型最终的验证集的分类准确率的平均数作为此下LOO-CV分类器的性能指标.相比于前面的K-CV,LOO-CV有两个明显的优点:


①a.每一回合中几乎所有的样本皆用于训练模型,因此最接近原始样本的分布,这样评估所得的结果比较可靠。


②b.实验过程中没有随机因素会影响实验数据,确保实验过程是可以被复制的。

但LOO-CV的缺点则是计算成本高,因为需要建立的模型数量与原始数据样本数量相同,当原始数据样本数量相当多时,LOO-CV在实作上便有困难几乎就是不显示,除非每次训练分类器得到模型的速度很快,或是可以用并行化计算减少计算所需的时间.

上小节数据集划分部分内容参考自博文[点击查看](http://blog.csdn.net/chl033/article/details/4671750)
