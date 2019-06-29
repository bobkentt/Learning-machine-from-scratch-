# 几个NLP任务

## 机器翻译
1. 机器翻译传统方法：概率模型和语法分析。现在常用方法基于深度学习seq2seq；
2. 回顾Transformer结构，引出self-attention；
3. Muti-head attention结构：用Muti-head思想有模型融合emsemble思想；从不同空间进行映射，再concat，取得更好效果

## 知识图谱
知识库可以做推理和表征关系


### 可以和bert一起联合训练模型


> vec_entity1 in S1

> vec_relationship in S2

> vec_entity1 in S1

> S1 == S2 or S1 != S2 

> vec_relationship * vec_entity1 = vec_entity2

> concat(word_vec,vec_entity2)

### 上下位词
手机是苹果手机的上位词，苹果手机是iphone7的上位词

## 摘要和纠错
* 摘要任务和文章改写比较像，是一个相对难的问题
* 新闻摘要相对于简单，因为文章中重点部分相对集中。
* 而其他问题难的原因是不确定那些信息是必要信息


文本纠错也可以用seq2seq模型来做，但关注的点不同

1. encoder layer要少忽略局部信息
2。 decoder layer要少见信息损失，尽量保留原始信息
