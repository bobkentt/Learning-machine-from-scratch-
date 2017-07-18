# 模型融合

## 模型融合（model ensemble）是什么
Ensemble Learnig 是一组individual learner的组合

> 如果individual learner同质，称为base learner

> 如果individual learner异质，称为component learner


## Why?
![](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/20170719-003836.png)

## Brief

简单说来，我们信奉几条信条

群众的力量是伟大的，集体智慧是惊人的

> Bagging

> 随机森林/Random forest


站在巨人的肩膀上，能看得更远

> 模型stacking

一万小时定律
> Adaboost

> 逐步增强树/Gradient Boosting Tree

### Bagging

过拟合了，bagging一下。

```
用一个算法
不用全部的数据集，每次取一个子集训练一个模型
分类：用这些模型的结果做vote
回归：对这些模型的结果取平均

```

### stacking

用多种predictor结果作为特征训练

![](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/20170719-004725.png)

### Adaboost
