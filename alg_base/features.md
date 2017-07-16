特征选择  
  ● 过滤型   sklearn.feature_selection.SelectKBest  （每个feature与label之间的相关性）
  ● 包裹型   sklearn.feature_selection.RFE （把feature排个序，再考察）
  ● 嵌入型  feature_selection.SelectFromModel Linear model，L1正则化（L1正则化可以截断特征，使W系数为0，L2对特征有缩放效应）
