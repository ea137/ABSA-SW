import pandas as pd
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

def unnest(y):
    o = []
    for i in y:
        o.extend(list(i))
    return np.array(o)


def MCC(true_labels,predictions):
    matthews_set = []

    for i in range(len(true_labels)):
        matt = matthews_corrcoef(true_labels[i],predictions[i])             
        matthews_set.append(matt)
    return matthews_set

def plot_MCC(matthews_set):
    plt.figure(figsize = (14,7))
    plt.bar(np.arange(len(matthews_set)),matthews_set)
    plt.title("MCC score per batch" ,size = 20)
    plt.xlabel("Batch" ,size = 20)
    plt.xticks(np.arange(len(matthews_set)))
    plt.ylabel("MCC Score")
    return plt.show()

def predict_bert_only(data_loader,model):
    model.eval() # putting model in evaluation mode

    predictions , true_labels = [], []


    for batch in data_loader:

        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            loss = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)


            predictions.append(np.array(np.argmax(loss.logits, axis= 1)))
            true_labels.append(np.array(b_labels))
    return predictions,true_labels

def find_spaces(str1):
    return len(str1.split())
def find_nth(sent, word, n):
    start = sent.find(word)
    while start >= 0 and n > 1:
        start = sent.find(word, start+1)
        n -= 1
    return start
dico_idx = {'negative' : 0,'positive' : 1, 'neutral':2}
def data_loader_no_extraction(filepath,find1 = './sentence',find2 = './aspectTerms/aspectTerm',find3="term" ):
    root = ET.parse(filepath).getroot()
    dico = {}
    for j in root.findall(find1):
        for i in j.findall(find2):
            key = j[0].text
            value = i.get(find3),  i.get('polarity'), int(i.get('from')), int(i.get('to'))
            if key in dico.keys():
                dico[key].append(value)
            else:
                dico[key] = [value]
    t = pd.DataFrame.from_dict(dico,orient = 'index')
    t['combined'] = [i[i != None] for i in t.values]
    t = t[['combined']].reset_index(level = 0)
    t_new = pd.DataFrame(columns=["sentence","aspect","polarity"])
    count = -1
    for sent,aspects in (t.values):
        for asp in (aspects):
            if not(asp[1] != 'positive' and asp[1] != 'negative'  and asp[1] != 'neutral'):
                count+=1
                #print([sent,asp[0],dico_idx[asp[1]]] )
                t_new.loc[len(t_new)] = [sent,asp[0],dico_idx[asp[1]]]               
    t_new["bert_input"] = (t_new["aspect"] +  " "+t_new["sentence"] + " "+t_new["aspect"])  
    return t_new


def tokedo(tttt,max_len):
    labels = tttt.polarity.values.astype('int64')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=1)  
    input_ids = []
    attention_masks = []
    for sent in tttt.bert_input:
        encoded_dico = tokenizer.encode_plus(sent,add_special_tokens=1,
                             max_length=max_len,
                             pad_to_max_length = True,
                             return_attention_mask=1,
                             return_tensors='pt',
                             truncation=True)

        input_ids.append(encoded_dico["input_ids"])
        attention_masks.append(encoded_dico['attention_mask'])

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids,attention_masks,labels)
    batch_size = dataset.tensors[0].size()[0]
    test_dataloader = DataLoader(dataset,
                                 sampler = RandomSampler(dataset),  
                                 batch_size=batch_size)
    return test_dataloader