import pandas as pd
import numpy as np
import re
import torch
import xml.etree.ElementTree as ET
from sklearn.metrics import f1_score,accuracy_score
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForMultipleChoice,BertTokenizer,BertForSequenceClassification, AdamW, BertConfig,get_linear_schedule_with_warmup
from helper import data_loader_no_extraction, find_nth, unnest
from First import DataForms
from sklearn.metrics import f1_score,accuracy_score






class ModelOne:
	def __init__(self,path,finds,test=False,train_perc=.9,epochs=3,saved_model = None,train_num= 1039):
		"""
			path: path of the dataset
			finds: xml tags to retrieve the data
			test: False if not test data, True otherwise
			saved_model: previously trained model
			train_num: number of training data to be considered.
	
		"""
		self.train_perc = train_perc
		self.epochs = epochs
		self.data = DataForms(path,finds,saved_model=None,test=True,train_num = train_num)
		self.max_len = self.data.max_len
		self.batch_size = self.data.batch_size
		self.data.dataForm() 
		self.train_dataloader = self.data.train_dataloader
		self.val_dataloader = self.data.val_dataloader
		self.dataloader = self.data.dataloader
		self.model = None
		
		if saved_model == None:
			self.saved_model = saved_model
		else:
			self.saved_model = torch.load(saved_model)

		self.predictions = None
		self.true_labels = None

                             
	def baseFormTraining(self):
		
		bert = BertForSequenceClassification.from_pretrained(
                        "bert-base-uncased", num_labels=self.max_len,
                        output_attentions = 0,output_hidden_states=1)
		optimizer = AdamW(bert.parameters(),lr = 5e-5,eps = 1e-8)
		scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=0,
                                           num_training_steps=len(self.train_dataloader) * self.epochs)
		for epoch in range(self.epochs):
			print("############ EPOCH ",epoch)
			bert.train()
			for step, batch in enumerate(self.train_dataloader):
				if step > 0:
				    print(f'  Batch %d  of  %d  ; bert loss : %.3f  '%(step, len(self.train_dataloader),loss.loss.item()))          
				else:
				    print(f'  Batch  %d   of   %d . '%(step, len(self.train_dataloader)))
				batch_input_ids,batch_input_mask,batch_labels = batch[0],batch[1],batch[2]

				bert.zero_grad() 
				loss = bert(batch_input_ids, 
				             token_type_ids=None, 
				             attention_mask=batch_input_mask, 
				             labels=batch_labels)
				loss.loss.backward()
				optimizer.step()
				scheduler.step()
		
			# evaluation 
			print("############ EVAL ")
			bert.eval()
			for batch in self.val_dataloader:
				batch_input_ids,batch_input_mask, batch_labels = batch[0],batch[1],batch[2]  
				with torch.no_grad():
				    loss = bert(batch_input_ids, 
				               token_type_ids=None, 
				               attention_mask=batch_input_mask,
				               labels=batch_labels)		
			print("f1 : ", f1_score(np.array(batch_labels),np.argmax(np.array(loss.logits), axis=1).flatten(), average="macro") )
			print("acc : ", accuracy_score(np.array(batch_labels),np.argmax(np.array(loss.logits), axis=1).flatten()))
		self.model = bert

	def baseFormPredicting(self):
		
		self.saved_model.eval()
		predictions , true_labels = [], []
		
		for batch in self.dataloader:
			batch_input_ids,batch_input_mask, batch_labels = batch[0],batch[1],batch[2]
			with torch.no_grad():
				loss = self.saved_model(batch_input_ids, token_type_ids=None, 
		                      attention_mask=batch_input_mask)
				predictions.append(np.array(np.argmax(loss.logits, axis= 1)))
				true_labels.append(np.array(batch_labels))

		self.predictions = unnest(predictions)
		self.true_labels = unnest(true_labels)
				
	def predictionStats(self):
	
		print(f"f1 score for all : %.3f, Accuracy for all : %.3f"%(f1_score(self.true_labels, self.predictions, average="macro"),
																accuracy_score(self.true_labels,self.predictions) ))    

	def saveModel(self,saving_path):
		torch.save(self.model, saving_path)

        



	
	
	
	
	
	
