# Xingzhi
A pretrained chinese text-to-image generation model for Xingzhi competition.

### model
 ![model](/model.png)
 -----------------------------
 
 ![model1](/model1.png)
 
 
### Env
|item|ver|
|----|----|
|OS|ubuntu 20.04|
|GPU|NVIDIA RTX 3090 24GB x6|
|python|3.10.4|
|cuda|11.3|
|torch|1.12.1|

### preprocess data
Converting the text to `tokens` by `cogview-data`, then packing the `ID` and `tokens` to `dataset/{train, val, test}.npy` using `make_dataset.py`.

### train && predict
We use 6 Nvidia RTX 3090.
Training process is about 20 days and inference is about 1 day.

**Important!! You must put the train and val images into the `dataset` floder, and rename them into `{train, val}` before train.**
Then 
```
bash scripts/train.sh
bash scripts/predict.sh
```

dataset: https://pan.baidu.com/s/1FaE8O25wq-1YWmVVHIPq8A passwd: zHvD

### Result
 ![result](/result.png)
 ----------------------
 
 ![result1](/result2.png)

