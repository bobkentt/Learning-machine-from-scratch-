# k-means算法
## 问题定义
Input：点集合set D，度量点之间距离的函数d，聚类后划分为k个set
Output：k个set

```
def k-means(D, d, k):
    # do something
```

其实是一个最优化目标函数问题，目标函数是：

```
# xi是D中的一个点
# c是xi属于的类的中间点
# d是度量点与点之间距离的函数
minimize(sum( min（ d(xi, c) ） ))

```

当k=1时，目标函数最优解在全部点的平均值。

## Lloyd's methold algorithm解决k-means问题
核心思想：
1. 找到点所属的类（计算点与各个中心点之间距离，找到距离最小的中心点，归类）；
2. 计算各个类的中心点（一个类中的点求平均）；
3. 重复步骤1和步骤2直到中心点不再更新。

#### 为什么Lloyd's methold（面试）？
目标函数有下界0，而且每一次迭代，目标函数都会减少。

因为单调递减有下界的函数一定是收敛的。

### 初始化
* random center of datapoint

* further travel最远遍历

* k-means++work well and provable guarantees.在工业上实际应用，K-means++本质上与k-means没有区别，只是在初始化阶段有独到的方法。

### random center of datapoint 运行过程

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/5.png)


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/6.png)


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/7.png)


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/8.png)


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/9.png)

最后算法收敛成，如下所示：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/10.png)

#### random initialization bad performance
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/11.png)


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/12.png)

发生bad performance 的概率（k个初始化中心点正好分在k个高斯分布中的概率）

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/13.png)


所以当k大一点的时候，random initialization是不行的。


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/14.png)

但是问题就是对抗噪声的能力差，如下图所示：


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/15.png)

### further travel最远遍历
初始化时使初始化中心尽可能的远。

1. 随机选取中心点c1，然后在剩下的点选取离c1最远的点c2；
2. 再选离c1和c2最远的点；
3. 依次类推直到选出k个点。

但是further travel最远遍历初始化方法，对抗噪声能力较差。

### k-means++初始化

1. 随机选择中心点c1；
2. 计算各个点离c1的距离；
3. 归一化，计算出每个点被选择做c2的概率
4. 安装计算出的概率，选出c2
5. 依次类推直到选出k个点。

在这个过程中有个概念d，d是点距离已经选择出的中心点的距离之和。

在实际中常根据d的α次方计算：
* α=0，随机初始化中心点；
* α=正无穷，最远初始化中心点；
* α=2，k-means++初始化；


## 时间复杂度

k-means++初始化时间复杂度是O(nkd)。n个点，k个类别，d表示点的维度。
因为每初始化一个点需要计算n各点到中心点的距离，所以复杂度是O(nd),循环k次所以是O(nkd)。


k-means的时间复杂度是：k-means每一轮循环的时间复杂度是O(nkd)。

因此k-means++初始化时间复杂度相对k-means不高。相当于多循环一次而已。

## 怎么选择k
1. 交叉验证法；
2. elbow's method：k越大分类效果越好，计算不同的k的时候的损失函数收敛的速度来解决；
