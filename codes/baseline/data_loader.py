import nibabel as nib
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from torch.nn.functional import normalize
from torch import FloatTensor
from PIL import Image
import cv2

# transform_image = transforms.Compose([transforms.Resize((256,256)),
# 									transforms.ToTensor(),
# 									transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


transform_image = transforms.Compose([transforms.Resize((256,256)),
									transforms.ToTensor()])

transform_label = transforms.Compose([transforms.Resize((256,256)),
									transforms.ToTensor() ])


class readDataset(Dataset):

	def __init__(self, images_path,labels_path, csv_path):

		self.images_path = images_path
		self.labels_path = labels_path
		self.df = pd.read_csv(csv_path)
		self.transform_image = transform_image		
		self.transform_label = transform_label
		
		
	def __len__(self):
		return len(self.df)


	def __getitem__(self, index):

		current_image = self.images_path+ "/" +str(self.df["image_id"][index])+".jpg"

		current_label = self.labels_path+ "/" +str(self.df["image_id"][index])+".png"
		
		image = cv2.imread(current_image)
		image=Image.fromarray(image)
		image = self.transform_image(image)
		

		label = cv2.imread(current_label,0)
		
		
		label = Image.fromarray(label)
		label = self.transform_label(label)
		

		return (image,label)




def trainDataLoaderFn(images_path,labels_path, csv_path, batch_size):

	trainSet = readDataset(images_path,labels_path, csv_path)
	return DataLoader(trainSet, batch_size=batch_size, shuffle=True)


def testDataLoaderFn(images_path,labels_path, csv_path,batch_size):

	testSet = readDataset(images_path,labels_path, csv_path)
	return DataLoader(testSet, batch_size = batch_size, shuffle=False)
