# Tf–idf 

特征TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语
料库中的其中一份文件的重要程度。字词的重要性随着它在文件中
出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

## 词频（Term Frequency，缩写为TF）

TF(t) = (词t在当前文中出现次数) / (t在全部文档中出现次数)

## 逆文档频率（Inverse Document Frequency，缩写为IDF）
IDF(t) = ln(总文档数/ 含t的文档数)
TF-IDF权重 = TF(t) * IDF(t)

[阮一峰的博客写的更清晰](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)
