### 代码运行环境说明
|项目|内容|
|----|----|
|OS|ubuntu 20.04|
|GPU|NVIDIA RTX 3090 24GB x6|
|python|3.10.4|
|cuda|11.3|
|torch|1.12.1|

### 数据预处理
使用`cogview-data`将文本转换为`tokens`，然后使用`make_dataset.py`将`ID`和`tokens`一起打包在`dataset/{train, val, test}.npy`

其中`val12.npy`为`val.npy`其中的12个样本，便于追踪试验结果
### train && predict
6张3090，大约需要训练20天左右，生成过程大约需要1天。

**重要！训练前，请将数据集训练图片与验证图片放在`dataset`下，并且分别将文件夹命名为`{train, val}`**
```
bash scripts/train.sh
bash scripts/predict.sh
```


