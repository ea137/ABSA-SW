## Steps associated with the implementation of the proposed model: 

1. Bert Training via Second.py
2. Creation of training and testing data using saved BERT model via First.py
3. Reduction of training and testing data using PCA via First.py
4. Training of LSTM via Third.py

______________________________


for ABSA 15 and 16 set finds to ["./Review/sentences/sentence",'./Opinions/Opinion',"target"]


for ABSA 14 set finds to ['./sentence','./aspectTerms/aspectTerm','term']

______________________________



The following example will be training on ABSA 15 of the restaurant domain (4 epochs on BERT, 7 on LSTM BLOCK)


EXAMPLE:
1. Run install.sh to download the library requirements
2. Create a folder and name it ./boom
3. Add the ABSA 15 dataset to the folder
4. Follow the steps below

Add following block to Second.py and delete it after running
<code>

path = "./boom/ABSA-15_Restaurants_Train_Final.xml"
finds = ["./Review/sentences/sentence",'./Opinions/Opinion',"target"]
pot = ModelOne(path,finds,epochs=4, train_num=1654)
pot.baseFormTraining()
pot.saveModel("./boom/absa15BERT")

</code>

Add following block to First.py and delete it after running
<code>

path = "boom/ABSA-15_Restaurants_Train_Final.xml"  # TO CHANGE
pot = DataForms(path, finds= ["./Review/sentences/sentence",'./Opinions/Opinion',"target"]
,saved_model = "./boom/absa15BERT" ,test=True)
pot.LSTMInputForm()
pot.saveForm("./boom/X","./boom/y")

</code>

Then

<code>

path = "boom/ABSA-15_Restaurants_Train_Final.xml"  # TO CHANGE
pot = DataForms(path, finds= ["./Review/sentences/sentence",'./Opinions/Opinion',"target"]
,saved_model = "./boom/absa15BERT" ,test=False)
pot.LSTMInputForm()
pot.saveForm("./boom/X","./boom/y")

</code>

Add following block to First.py and delete it after running
<code>

path = "boom/ABSA-15_Restaurants_Train_Final.xml"  # TO CHANGE
pot = DataForms(path, finds= ["./Review/sentences/sentence",'./Opinions/Opinion',"target"]
,saved_model = "./boom/absa15BERT" )
pot.ReduceLSTMPCA("./boom/X False.pkl","./boom/X True.pkl")

</code>

Add following block to Third.py and delete it after running
<code>

pot = ModelTwo("./boom/X False_reduced.pkl","./boom/X True_reduced.pkl","./boom/y False.pkl","./boom/y True.pkl"
,train_perc=.9,epochs=7,batch_size = 32)
pot.upperFormTraining()
pot.saveModel("./boom/Final_model.h5")

</code>




