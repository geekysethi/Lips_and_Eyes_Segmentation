import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

train_df = pd.read_csv("C:\\Users\\ashishHP\Desktop\\fynd_assignment\\tensorboard_data\\train_iou.csv")

print(train_df.head())
train_iou = train_df["Value"].tolist()
step = train_df["Step"].tolist()

plt.figure()
plt.plot(step,train_iou)
plt.title("Training IOU")
plt.xlabel("Epoch")
plt.ylabel("IOU")
plt.show()
