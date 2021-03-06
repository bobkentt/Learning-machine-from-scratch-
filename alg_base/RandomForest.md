# 随机森林 （Random Forest）

## 工作机制
1. 从原始训练数据集中，应⽤自助采样法(bootstrap)⽅法有放回地随机抽取k个新的⾃助样本集，
并由此构建k棵分类回归树，每次未被抽到的样本组成了Ｋ个袋外数据（out-ofbag,BBB）；
2. 设有n个特征，则在每⼀棵树的每个节点处随机抽取m个特征，从子集中选择最优特征划分；
3. 每棵树最⼤限度地⽣长， 不做任何剪裁；
4. 将⽣成的多棵树组成随机森林， ⽤随机森林对新的数据进⾏分类，
分类结果按树分类器投票多少⽽定。

## 性能
与Bagging对比，Bagging使用却限定性决策树，在选择特征划分时对所有结点进行考察，随机森林只对部分特征进行考察，所以RF性能高于Bagging。
