import os
import shutil
import time
import sys
sys.path.append("models/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms

from utils import progress_bar

from model import UNet
from loss_functions import DiceLoss, calc_IOU

class Trainer():

	def __init__(self,config,trainLoader,validLoader):
		
		self.config = config
		self.trainLoader = trainLoader
		self.validLoader = validLoader
		

		self.numTrain = len(self.trainLoader.dataset)
		self.numValid = len(self.validLoader.dataset)
		
		self.saveModelDir = str(self.config.save_model_dir)+"/"
		
		self.bestModel = config.bestModel
		self.useGpu = self.config.use_gpu


		self.net = UNet()


		if(self.config.resume == True):
			print("LOADING SAVED MODEL")
			self.loadCheckpoint()

		else:
			print("INTIALIZING NEW MODEL")

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.net = self.net.to(self.device)

	

		self.totalEpochs = config.epochs
		

		self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4)
		self.loss = DiceLoss()

		self.num_params = sum([p.data.nelement() for p in self.net.parameters()])
		
		self.trainPaitence = config.train_paitence
		

		if not self.config.resume:																																																																																																																																																																																		# self.freezeLayers(6)
			summary(self.net, input_size=(3,256,256))
			print('[*] Number of model parameters: {:,}'.format(self.num_params))
			self.writer = SummaryWriter(self.config.tensorboard_path+"/")

		
		
		

	def train(self):
		bestIOU = 0

		print("\n[*] Train on {} sample pairs, validate on {} trials".format(
			self.numTrain, self.numValid))
		

		for epoch in range(0,self.totalEpochs):
			print('\nEpoch: {}/{}'.format(epoch+1, self.totalEpochs))
			
			self.trainOneEpoch(epoch)

			validationIOU = self.validationTest(epoch)

			print("VALIDATION IOU: ",validationIOU)

			# check for improvement
			if(validationIOU > bestIOU):
				print("COUNT RESET !!!")
				bestIOU=validationIOU
				self.counter = 0
				self.saveCheckPoint(
				{
					'epoch': epoch + 1,
					'model_state': self.net.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'best_valid_acc': bestIOU,
				},True)

			else:
				self.counter += 1
				
			
			if self.counter > self.trainPaitence:
				self.saveCheckPoint(
				{
					'epoch': epoch + 1,
					'model_state': self.net.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'best_valid_acc': validationIOU,
				},False)
				print("[!] No improvement in a while, stopping training...")
				print("BEST VALIDATION IOU: ",bestIOU)

				return None

		
	def trainOneEpoch(self,epoch):
		self.net.train()
		train_loss = 0
		total_IOU = 0
		
		for batch_idx, (images,targets) in enumerate(self.trainLoader):


			images = images.to(self.device)
			targets = targets.to(self.device)

			
	
			self.optimizer.zero_grad()

			outputMaps = self.net(images)
			
			loss = self.loss(outputMaps,targets)
			

			
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()

			current_IOU = calc_IOU(outputMaps,targets)
			total_IOU += current_IOU
			
			del(images)
			del(targets)

			progress_bar(batch_idx, len(self.trainLoader), 'Loss: %.3f | IOU: %.3f'
		% (train_loss/(batch_idx+1), current_IOU))
		self.writer.add_scalar('Train/Loss', train_loss/batch_idx+1, epoch)
		self.writer.add_scalar('Train/IOU', total_IOU/batch_idx+1, epoch)
		
		


	def validationTest(self,epoch):
		self.net.eval()
		validationLoss = []
		total_IOU = []
		with torch.no_grad():
			for batch_idx, (images,targets) in enumerate(self.validLoader):
				
				
				
				images = images.to(self.device)
				targets = targets.to(self.device)


				outputMaps = self.net(images)

				loss = self.loss(outputMaps,targets)


				currentValidationLoss = loss.item()
				validationLoss.append(currentValidationLoss)
				current_IOU = calc_IOU(outputMaps,targets)
				total_IOU.append(current_IOU)

			
				# progress_bar(batch_idx, len(self.validLoader), 'Loss: %.3f | IOU: %.3f' % (currentValidationLoss), current_IOU)


				del(images)
				del(targets)

		meanIOU = np.mean(total_IOU)
		meanValidationLoss = np.mean(validationLoss)
		self.writer.add_scalar('Validation/Loss', meanValidationLoss, epoch)
		self.writer.add_scalar('Validation/IOU', meanIOU, epoch)
		
		print("VALIDATION LOSS: ",meanValidationLoss)
				
		
		return meanIOU



	def test(self,dataLoader):

		self.net.eval()
		testLoss = []
		total_IOU = []

		total_outputs_maps = []
		total_input_images = []
		
		with torch.no_grad():
			for batch_idx, (images,targets) in enumerate(dataLoader):

				images = images.to(self.device)
				targets = targets.to(self.device)


				outputMaps = self.net(images)

				
				loss = self.loss(outputMaps,targets)

				testLoss.append(loss.item())
				current_IOU = calc_IOU(outputMaps,targets)
				
				total_IOU.append(current_IOU)
				
				total_outputs_maps.append(outputMaps.cpu().detach().numpy())


				# total_input_images.append(transforms.ToPILImage()(images))
				
				total_input_images.append(images.cpu().detach().numpy())

				del(images)
				del(targets)
				break

		meanIOU = np.mean(total_IOU)
		meanLoss = np.mean(testLoss)
		print("TEST IOU: ",meanIOU)
		print("TEST LOSS: ",meanLoss)	

		return total_input_images,total_outputs_maps
		

		
	def saveCheckPoint(self,state,isBest):
		filename = "model.pth"
		ckpt_path = os.path.join(self.saveModelDir, filename)
		torch.save(state, ckpt_path)
		
		if isBest:
			filename = "best_model.pth"
			shutil.copyfile(ckpt_path, os.path.join(self.saveModelDir, filename))

	def loadCheckpoint(self):

		print("[*] Loading model from {}".format(self.saveModelDir))
		if(self.bestModel):
			print("LOADING BEST MODEL")

			filename = "best_model.pth"

		else:
			filename = "model.pth"

		ckpt_path = os.path.join(self.saveModelDir, filename)
		print(ckpt_path)
		
		if(self.useGpu==False):
			self.net=torch.load(ckpt_path, map_location=lambda storage, loc: storage)


			

		else:
			print("*"*40+" LOADING MODEL FROM GPU "+"*"*40)
			self.ckpt = torch.load(ckpt_path)
			self.net.load_state_dict(self.ckpt['model_state'])

			self.net.cuda()
