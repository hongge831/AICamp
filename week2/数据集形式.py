# 加载数据分析常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
% matplotlib inline
img = Image.open('/mnt/datasets/train_images/neutral_sliced/neutral-25776.jpg')
plt.imshow(img)
img = np.array(img)
print(img.shape)