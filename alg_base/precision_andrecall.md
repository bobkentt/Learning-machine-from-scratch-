# 精确率与召回率，RoC曲线与PR曲线

在机器学习的算法评估中，尤其是分类算法评估中，我们经常听到精确率(precision)与召回率(recall)，RoC曲线与PR曲线这些概念，那这些概念到底有什么用处呢？

## 精确率与召回率
首先，我们需要搞清楚几个拗口的概念：

1. TP, FP, TN, FN
* True Positives,TP：预测为正样本，实际也为正样本的特征数
* False Positives,FP：预测为正样本，实际为负样本的特征数
* True Negatives,TN：预测为负样本，实际也为负样本的特征数
* False Negatives,FN：预测为负样本，实际为正样本的特征数


### 精确率（Precision）:

```
P = TP / (TP+FP)

```

### 召回率(Recall)又称为查全率:


```
R = TP / (TP+FN)

```

## ROC曲线和PR曲线

![pic](http://images2015.cnblogs.com/blog/1042406/201610/1042406-20161024164359046-1869944207.png)
