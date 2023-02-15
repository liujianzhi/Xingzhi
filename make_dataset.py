import numpy as np
from tqdm import tqdm
import os
import pandas
file = np.load('dataset-test.npy', allow_pickle=True)
        
# train_outputfile = []
# val_outputfile = []
test_outputfile = []
for id, text in tqdm(file):
    text = np.array(text)
    text_tokens = np.ones([256]) * 50001
    text = text[np.array(text) > 0] - 16384
    text_tokens[:len(text)] = text
    list = [id, text_tokens]
    test_outputfile.append(list)

# np.save('dataset/train.npy', train_outputfile)
# np.save('dataset/val.npy', val_outputfile)
np.save('dataset/test.npy', test_outputfile)