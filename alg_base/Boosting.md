# Boosting
## 工作机制
1. 先初始化训练集D1，训练出一个基学习器L1；
2. 根据基学习器L1的表现对训练集D1样本分布进行调整，生成样本集D2，D2使得基学习器之前做错的训练样本在后续受到更多关注；
3. 基于调整后的D2,训练下一个基学习器L2；
4. 重复以上过程，直到学习出T个学习器（T为超参数）；
5. 最终预测时对这T个基学习器加权结合；

从偏差-方差分解的角度看，Boosting更关注降低偏差。因此Boosting能基于泛化性能相当弱的学习期构建出很强的集成。

## 关于xgboost的理解和推导
> 陈天奇大神关于Bossted tree的文章[Boosted tree](http://www.52cs.org/?p=429)

## More介绍XGBoost的文章
一篇写的很好的xGBost文章，我就不用再写一遍了。链接在下面：

> [点击跳转:xgboost原理](http://blog.csdn.net/a819825294/article/details/51206410)

> [点击跳转:从ID3到XGBoost](http://www.jianshu.com/p/41dac1f6b0d2)

> [xgboost作者写的关于xgboost](https://courses.cs.washington.edu/courses/cse546/14au/slides/oct22_recitation_boosted_trees.pdf)
