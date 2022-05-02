import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch
import pickle
from helper import data_loader_no_extraction, find_nth
from sklearn.decomposition import PCA



class DataForms:
	def __init__(self,path,finds,saved_model=None,test=False,batch_size =32,max_len=76,train_num=1039):
	"""
	path: path of the dataset
	finds: xml tags to retrieve the data
	saved_model: previously trained model
	max_len: maximum sentence length
	train_num: number of training data to be considered.
	
	"""
		self.train_num = train_num
		self.max_len = max_len
		self.test = test
		self.batch_size = batch_size
		if saved_model == None:
			self.saved_model = saved_model
		else:
			self.saved_model = torch.load(saved_model)
		self.test = test
		self.dico_idx =  {'negative' : 0,'positive' : 1,'neutral' : 2}
		self.considered_df_category = data_loader_no_extraction(path,find1=finds[0],find2 =finds[1],find3 =finds[2])
		
		self.train_dataloader = None
		self.val_dataloader = None
		self.dataloader = None
		self.X_test = None
		self.y_test = None
		
	def reorg(self):
		self.considered_df_category['sentence'] = self.considered_df_category['sentence'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
		self.considered_df_category["sentence"]= self.considered_df_category["sentence"].apply(lambda x : x.lower())                
		self.considered_df_category['aspect'] = self.considered_df_category['aspect'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
		self.considered_df_category["aspect"]= self.considered_df_category["aspect"].apply(lambda x : x.lower())            
		self.considered_df_category['bert_input'] = self.considered_df_category['bert_input'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
		self.considered_df_category["bert_input"]= self.considered_df_category["bert_input"].apply(lambda x : x.lower())   
		

	def dataForm(self,data=None):
		self.reorg()
		try:
			if data ==None:
				data = self.considered_df_category
		except:
			pass
		
		#data = self.considered_df_category
		labels = data.polarity.values.astype('int64')
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=1)
		input_ids = []
		attention_masks = []
		for sent in data.bert_input:
			encoded_dico = tokenizer.encode_plus(sent,add_special_tokens=1,
		    									max_length=self.max_len,
		    									pad_to_max_length = 1,
		   										return_attention_mask=1,
		  										return_tensors='pt',
		 										truncation=True)
			input_ids.append(encoded_dico["input_ids"])
			attention_masks.append(encoded_dico['attention_mask'])
		input_ids = torch.cat(input_ids)
		attention_masks = torch.cat(attention_masks)
		labels = torch.tensor(labels)
		dataset = TensorDataset(input_ids,attention_masks,labels)
		train_size = int( .9*len(dataset) )
		val_size = len(dataset) - train_size
		train_data, val_data = random_split(dataset,[train_size,val_size])
		self.train_dataloader = DataLoader(train_data,
	                         sampler = RandomSampler(train_data),  
	                         batch_size=self.batch_size)
		self.val_dataloader = DataLoader(val_data,
	                         sampler = RandomSampler(val_data),
	                         batch_size=self.batch_size)
		self.dataloader = DataLoader(dataset,
                         sampler = RandomSampler(dataset),  
                         batch_size=dataset.tensors[0].size()[0])
	
	def LSTMInputForm(self):
		considered_df_category_test = self.considered_df_category[self.train_num:]         
		considered_df_category = self.considered_df_category[:self.train_num]
		if self.test:
			tttt = considered_df_category_test
		else:
			tttt = considered_df_category
		self.dataForm(data = tttt)
		
		for step, batch in enumerate(self.dataloader):
			batch_input_ids,batch_input_mask, batch_labels = batch[0],batch[1],batch[2]
			with torch.no_grad():
				loss = self.saved_model(batch_input_ids, token_type_ids=None, 
				          			attention_mask=batch_input_mask)
			X_test = []
			y_test = batch_labels.detach().numpy()
			for i in range(len(loss.hidden_states)):
				X_test.append(loss.hidden_states[i].detach().numpy().reshape(-1,self.max_len*768,1) ) 
		self.y_test = np.array(y_test)
		self.X_test = np.array(X_test)
		
	def ReduceLSTMPCA(self,trainPath,TestPath, n_components=1039,):

		with open(trainPath,'rb') as outfile:
			X_Train = pickle.load(outfile)
		with open(TestPath,'rb') as outfile:
			X_Test = pickle.load(outfile)
			
		X_train_reduced = np.ones((13,X_Train.shape[1],n_components))
		X_test_reduced = np.ones((13,X_Test.shape[1],n_components))
		if len(X_Train.shape)==4:
			X_Train = X_Train.reshape(X_Train.shape[:-1])
		if len(X_Test.shape)==4:
			X_Test = X_Test.reshape(X_Test.shape[:-1])

		for i in range(13):
			print(i)
			pca = PCA(n_components=n_components)
			pca.fit(X_Train[i])
			X_train_reduced[i] = pca.transform(X_Train[i])
			X_test_reduced[i] = pca.transform(X_Test[i])

		with open(trainPath[:-4] + "_reduced" + trainPath[-4:], 'wb') as outfile:
			pickle.dump(X_train_reduced, outfile, pickle.HIGHEST_PROTOCOL)
		with open(TestPath[:-4] + "_reduced" + TestPath[-4:], 'wb') as outfile:
			pickle.dump(X_test_reduced, outfile, pickle.HIGHEST_PROTOCOL)
	
   
		return {"train":X_train_reduced, "test":X_test_reduced}
		
		
	def saveForm(self,X_path,y_path):
		with open(X_path+" " + str(self.test) + ".pkl", 'wb') as outfile:
			pickle.dump(self.X_test, outfile, pickle.HIGHEST_PROTOCOL)
		with open(y_path+" " + str(self.test) + ".pkl", 'wb') as outfile:
			pickle.dump(self.y_test, outfile, pickle.HIGHEST_PROTOCOL)
	def IsTest(self):
		print(self.test)
		return test
		

	
	
	
