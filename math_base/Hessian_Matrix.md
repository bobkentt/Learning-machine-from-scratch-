# Hessian Matrix

Hessian矩阵：在数学中，海森矩阵（Hessian matrix 或 Hessian）
是一个多变量实值函数的二阶偏导数组成的方阵，假设有一实数函数 f(x_1,x_2,…,x_n) ，
如果 f 所有的二阶偏导数都存在，那么 f 的Hessian矩阵的第 ij 项是 
![](http://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_i%5Cpartial+x_j%7D)，
即Hessian矩阵为

![](http://www.zhihu.com/equation?tex=H%28f%29%3D+%5Cleft%5B+%5Cbegin%7Bmatrix%7D+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_1%5E2%7D+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_1%5Cpartial+x_2%7D+%26+%5Cldots+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_1%5Cpartial+x_n%7D+%5C%5C+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_2%5Cpartial+x_1%7D+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_2%5E2%7D+%26+%5Cldots+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_2%5Cpartial+x_n%7D+%5C%5C+%5Cvdots+%26+%5Cvdots+%26+%5Cddots+%26+%5Cvdots+%5C%5C+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_n%5Cpartial+x_1%7D+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_n%5Cpartial+x_2%7D+%26+%5Cldots+%26+%5Cfrac%7B%5Cpartial%5E2f%7D%7B%5Cpartial+x_n%5E2%7D+%5Cend%7Bmatrix%7D+%5Cright%5D)

如果函数 f 在定义域 D 内的每个二阶导数都是连续函数，那么 f 的海森矩阵在 D 区域内为对称矩阵。

极值点：基于Hessian矩阵 H 我们可以判断多元函数的极值情况：

* 如果 H 是正定矩阵，则临界点一定是局部极小值点。
* 如果 H 是负定矩阵，则临界点一定是局部极大值点。
* 如果行列式 |H|=0 ，需要更高阶的导数来帮助判断。
* 在其余情况下，临界点不是局部极值点。

实对称矩阵可对角化：若 A 是实对称矩阵，
则存在正交矩阵 Q 使得 
![](http://www.zhihu.com/equation?tex=QAQ%5ET%3D%5CLambda%3Ddiag%28%5Clambda_1%2C%E2%80%A6%2C%5Clambda_n%29)，
其中 
![](http://www.zhihu.com/equation?tex=%5Clambda_i)
是矩阵 A 的特征值。

若 A 可逆（即非奇异），则每个 
![](http://www.zhihu.com/equation?tex=%5Clambda_i)
都非零且 
![](http://www.zhihu.com/equation?tex=%5Clambda_i%5E%7B-1%7D)
是 
![](http://www.zhihu.com/equation?tex=A%5E%7B-1%7D)
的特征值， i=1,2,…,n 。


