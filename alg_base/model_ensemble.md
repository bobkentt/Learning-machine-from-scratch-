# 模型融合
## 模型融合（model ensemble）是什么
Ensemble Learnig 是一组individual learner的组合：
* 如果individual learner同质，称为base learner
* 如果individual learner异质，称为component learner
## 集成学习分类
根据个体学习器间存在的关系，目前集成学习大致可以分为两大类：
* 个体学习器存在强依赖关系，必须串行生成的序列化方法(Boosting)；
* 个体学习器不存在强依赖关系，可以并行生成（Bagging、Random Forest）；
## 模型融合为什么能有效果
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170719-003836.png)
## 模型融合信奉几条信条
简单说来，我们信奉几条信条:
1. 群众的力量是伟大的，集体智慧是惊人的，包括：Bagging和随机森林（Random forest)
2. 站在巨人的肩膀上，能看得更远:模型stacking
3. 一万小时定律:Adaboost和逐步增强树（Gradient Boosting Tree）
### Bagging
过拟合了，bagging一下。

用一个算法:
* 不用全部的数据集，每次取一个子集训练一个模型
* 分类：用这些模型的结果做vote
* 回归：对这些模型的结果取平均

用不同的算法:
* 用这些模型的结果做vote 或 求平均
### stacking
用多种predictor结果作为特征训练
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/20170719-004725.png)
### Adaboost
[参考博文，不赘述](http://blog.csdn.net/google19890102/article/details/46376603)
