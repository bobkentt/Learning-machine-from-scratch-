# RMSE and MAE
评分预测的预测准确度一般通过均方根误差(RMSE)和平均绝对误差(MAE)计算。
 
records[i] = [u,i,rui,pui]

* 对于测试集中的一个用户u和物品i；
* rui表示用户u对物品i的实际评分；
* pui表示推荐算法给出的预测评；

如下代码可以计算RMSE和MAE：

```
    def RMSE(records):
        return math.sqrt(\
             sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records])\
             / float(len(records)))
    def MAE(records):
        return sum([abs(rui-pui) for u,i,rui,pui in records])\
             / float(len(records))
```
