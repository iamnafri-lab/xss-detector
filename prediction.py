import tensorflow as tf
from utils import GeneSeg
import numpy as np
import json
import pickle
import pandas as pd
from processing import batch_generator, dataTest,build_dataset
from keras.preprocessing.sequence import pad_sequences
import random
model = tf.keras.models.load_model("file/CNN_LSTM4_9_model")

def to_index(data):
        d_index=[]
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index

def pre(x):
    datas_index=[to_index(data) for data in x]
    datas_index=pad_sequences(datas_index,value=-1)
    rand=random.sample(range(len(datas_index)),len(datas_index))
    datas=[datas_index[index] for index in rand]
    appendingData=[]
    counter=755-len(datas[0])
    for i in range(counter):
        appendingData.append(-1)
    appendingDataCopy=appendingData
    for j in range(len(datas)):
        for i in range(len(datas[j])):
            appendingDataCopy.append(datas[j][i])
        datas[j]=appendingDataCopy
        appendingDataCopy=[]
        for w in range(counter):
            appendingDataCopy.append(-1)
    w=np.asarray(datas,dtype='int32')
    return w

def create_file(w):
    with open("PredictionDataTest.csv","w") as f:
                for i in range(len(w)):
                    data_line=str(w[i].tolist())+"\n"
                    f.write(data_line)

def data_generator(data_dir):
    df = tf.data.TextLineDataset([data_dir])
    for line in df:
        try:
            [data, label] = tf.strings.split(line, b"|").numpy()
        except:
            [data] = tf.strings.split(line, b"|").numpy()
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array([])
        yield (data, label)

def batch_generator(datas_dir,datas_size,batch_size,embeddings,reverse_dictionary,train=True):
    batch_data = []
    batch_label = []
    generator=data_generator(datas_dir)
    n=0
    while True:
        for i in range(batch_size):
            data,label=next(generator)
            data_embed = []
            for d in data:
                if d != -1:
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(embeddings["UNK"]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n+=1
            if not train and n==datas_size:
                break
        if not train and n == datas_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data),np.array(batch_label))
            batch_data = []
            batch_label = []

def predict(w):
    test=batch_generator("PredictionDataTest.csv",len(w),1,embeddings,reverse_dictionary,train=False)
    arr = model.predict(test)
    return arr

vec_dir="file/word2vec.pickle"
with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
dictionary=word2vec["dictionary"]
embeddings = word2vec["embeddings"]
reverse_dictionary = word2vec["reverse_dictionary"]
train_size=word2vec["train_size"]
test_size=word2vec["test_size"]
dims_num = word2vec["dims_num"]
input_num =word2vec["input_num"]



print("\n____________________________________________________________________________________\n")
line = input("Enter Input to Predict : ")
print("\n____________________________________________________________________________________\n")
x=GeneSeg(line)
pre_file = pre(x)
create_file(pre_file)
result = predict(pre_file)
yes=0
no=0
for x in range(len(result)):
    yes += result[x][0]
    no += result[x][1]
print("yes : ", yes/len(result) , " No : ", no/len(result))


