import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

train_df = pd.read_csv("C:\\Users\\ashishHP\Desktop\\fynd_assignment\\tensorboard_data\\train_loss.csv")
validation_df = pd.read_csv("C:\\Users\\ashishHP\Desktop\\fynd_assignment\\tensorboard_data\\validation_loss.csv")

validation_loss = validation_df["Value"].tolist()
train_loss = train_df["Value"].tolist()
step = train_df["Step"].tolist()

plt.figure()
plt.plot(step[0:25],train_loss[0:25])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure()
plt.plot(step[0:25],validation_loss)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
