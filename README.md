# beads_Classification  

整個流程大概是這個樣子:  
1.指定資料夾  
2.讀取指定資料夾中檔名是".tiff"的圖檔  
3.再一系列前處理之後，label並切下圖中的珠子  
4.分類  

版本是tf_2_10，想用別的可以自己改
主程式: Total_4.py  
單顆beads分類程式: single_tf_2_10.py
多顆初步分類程式: resnet.py
有沒有接上的分類程式: monochrome_classifier.py

以上

*補充說明:
1. 單顆的是普通的CNN，但我有亂改
2. 多顆的是基底是ResNet18，但我有亂改
3. 有沒有接上的程式，基底是ResNet50，但我裡面寫ResNet34記得不要被我騙到

如果還有其他的會再補充， 2024/08/07

