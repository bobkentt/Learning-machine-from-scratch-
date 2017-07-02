
# k-means algorithm

## Model定义
Input：A set S of n points,also a distance/dissimilarity来度量每个点的距离，例如d(x,y)

Output:output a parttion of the data

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/1.png)

### coputation compexity
* NP hard even for k=2 or d=2

### Easy case when k=1


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/2.png)

求导可解，


![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/3.png)

其实右边还有一个多项式，从上面可以看出：

当C=u时第一个u-c=0,

同时summary(Xi-u)也是最小的

## 求解方法
### Lloyd's methold algorithm
Lloyd's methold可以解决，Lloyd's methold如下：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/4.png)

#### 为什么Lloyd's methold（面试）？
因为每一次迭代，目标函数都会减少，而且目标函数是一个非负的函数。

因为单调递减有下界的函数一定是收敛的

#### initialization of Lloyd's methold

* random center of datapoint

* further travel最远遍历

* k-means++work well and provable guarantees.在工业上实际应用，K-means++本质上与k-means没有区别，只是在初始化阶段有独到的方法

initialization of Lloyd's methold random

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

发生bad performance 的概率

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/13.png)


所以当k大一点的时候，random initialization是不行的。
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/14.png)

但是问题就是对抗噪声的能力差，如下图所示：
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/15.png)
