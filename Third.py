import keras
import keras.backend as K
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Dropout, concatenate, Conv2D, TimeDistributed, Bidirectional, Flatten, Conv1D, Reshape                
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.utils.np_utils import to_categorical 
import pickle
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from keras.models import load_model

class ModelTwo:
	def __init__(self,XTrain_path,XTest_path,
	yTrain_path,yTest_path,train_perc=.9,epochs=7,batch_size = 32,n_components=1039,saved_model =None):
		"""
			path: path of the dataset
			*_path: training/testing data
			n_components: components considered in PCA reduction // don't change (otherwise change it to its corresponding value in First.py)
			saved_model: previously trained model
			
	
		"""
		
		self.train_perc = train_perc
		self.epochs = epochs
		self.batch_size = batch_size
		
		with open(XTrain_path, 'rb') as infile:
			X_train_reduced = pickle.load(infile)
		with open(XTest_path, 'rb') as infile:
			X_test_reduced = pickle.load(infile)
			
		with open(yTrain_path, 'rb') as infile:
			y_train = pickle.load(infile)
		with open(yTest_path, 'rb') as infile:
			y_test = pickle.load(infile)
		
		self.model = None
		
		if saved_model == None:
			self.saved_model = saved_model
		else:
			self.saved_model = load_model(saved_model)
		
		self.n_components = n_components
		
		self.X_train_reduced= X_train_reduced.reshape(13,-1,self.n_components,1)           
		self.X_test_reduced= X_test_reduced.reshape(13,-1,self.n_components,1)       
		self.y_test = to_categorical(y_test)
		self.y_train = to_categorical(y_train)
		
	def block(self,name,intype=1):
		inputs = keras.Input(shape = [self.n_components ,1])
		with K.name_scope(name):     
		    x1 = LSTM(32, return_sequences=True)(inputs) 
		    #x1 = Dropout(0.4)(x1)
		    x1 = Bidirectional(LSTM(32, return_sequences=0))(x1)
		    x1 = Dropout(.4)(x1)
		    
		return x1,inputs
	
		
	def upperFormTraining(self):
		lstm1,input1 = self.block("h1")
		lstm2,input2 = self.block("h2")
		lstm3,input3 = self.block("h3")
		lstm4,input4 = self.block("h4")
		lstm5,input5 = self.block("h5")
		lstm6,input6 = self.block("h6")
		lstm7,input7 = self.block("h7")
		lstm8,input8 = self.block("h8")
		lstm9,input9 = self.block("h9")
		lstm10,input10 = self.block("h10")
		lstm11,input11 = self.block("h11")
		lstm12,input12 = self.block("h12")
		lstm13,input13 = self.block("h13")
		
		x= concatenate([lstm1,lstm2,lstm3,lstm4,
                lstm5,lstm6,lstm7,lstm8,lstm9,
                lstm10, lstm11, lstm12, lstm13])

		out = Dense(3,activation='softmax')(x)
		lstm = Model([input1,input2,input3,input4 ,
              input5 ,input6,input7 ,input8 ,input9 ,
              input10,input11,input12,input13] ,out)
		opt = keras.optimizers.Adam(learning_rate=.01)
		lstm.compile( optimizer=opt,loss='categorical_crossentropy', metrics = ["accuracy"])
		inputsTrain = list(self.X_train_reduced)
		hist = lstm.fit(inputsTrain,                         
		self.y_train,epochs=self.epochs,batch_size = self.batch_size , validation_split=.3)
		self.model = lstm
		return self.model,hist
	def upperFormPredicting(self):
		pred_test = self.saved_model.predict(list(self.X_test_reduced))
		self.pred_test = np.argmax(pred_test,1)
		
		pred_train = self.saved_model.predict(list(self.X_train_reduced))
		self.pred_train = np.argmax(pred_train,1)

		
		
	def predictionStats(self):
		
		print(np.argmax(self.y_test,1).shape, self.pred_test.shape)
		print(np.argmax(self.y_train,1).shape, self.pred_train	.shape)
		print(f"accuracy score for test : %.3f, for train : %.3f"%(accuracy_score(np.argmax(self.y_test,1), self.pred_test ), accuracy_score(np.argmax(self.y_train,1), self.pred_train) ))             
		
		print(f"f1 score for test : %.3f, for train : %.3f"%(f1_score(np.argmax(self.y_test,1), self.pred_test, average = "macro"), f1_score(np.argmax(self.y_train,1), self.pred_train, average= "macro")  ))             
		
		
  
		
	def saveModel(self,saving_path):
		self.model.save(saving_path)


pot = ModelTwo("./boom/X False_reduced.pkl","./boom/X True_reduced.pkl","./boom/y False.pkl","./boom/y True.pkl"
,train_perc=.9,epochs=7,batch_size = 32)
pot.upperFormTraining()
pot.saveModel("./boom/Final_model.h5")

		
		
		
				

