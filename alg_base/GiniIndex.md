# 基尼系数(Gini Index)
写了一个简单的计算计算基尼系数的程序。

```
# 数据集D
# x = [1,0,1.0,3,4]
# D = [(x1,y1),(x2,y2),(x3,y3)...(xn,yn)]

def Gini():
    count = 0
    clf_count = {}
    for d in D:
        count++
        if d[1] not in clf_count:
            clf_count[d[1]] = 0
        clf_count[d[1]]++
    
    if count == 0:
        return false, 0     

    sum = 0
    count_squ = count * count
    for cc in clf_count:
        sum = sum + (cc*cc/count_squ)
    return true, (1 - sum)

# a是数据集中的第i个属性
def Gini_index(i):
    # 假设属性a可以取1,2,3三个值
    a = [1,2,3]
    # 假设样本集中，属性a取1，2，3对应样本数分别为10，20，30
    a_col = {1:10, 2:20, 3:30}
    count = 10 + 20 + 30
    sum = 0
    for c,ck in a_col.items():
        dv = []
        for d in D:
            if d[i] == ck:
                dv.append(d)
        sum = sum + (c /count * Gini(dv))
```

