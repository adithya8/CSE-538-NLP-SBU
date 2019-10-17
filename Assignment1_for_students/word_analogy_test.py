import os, sys
import pickle
import numpy as np
import pdb
from scipy import spatial
import subprocess

if(len(sys.argv)>1):
    if sys.argv[1] == 'nce':
        loss_model = 'nce'
    else:
        loss_model = 'cross_entropy'

model_path = './models/'
if(len(sys.argv)>2):
    model_filepath = sys.argv[2]
else:
    model_filepath = input('Custom input (Press enter for default):')
    model_filepath = model_filepath if(model_filepath != "") else os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

if not os.path.exists('./word_analogy_pred'):
    os.mkdir('./word_analogy_pred')
    print ("Created a path: %s"%('./word_analogy_pred'))
outputFile = './word_analogy_pred'+'/word_analogy_test_predictions_'+loss_model+'.txt'
if(len(sys.argv)>3):
    outputFile = str(sys.argv[3])

f = open('./word_analogy_test.txt','r')
a = f.readlines()
f.close()    

directionVec = []
wordVec = []
words = []
for i in a:
    keys = i.strip().split("||")[0].split('"')
    values = i.strip().split("||")[1].split('"')
    r = []
    for j in keys:
        if (':' in j):
            key = j.split(":")
            r.append(embeddings[dictionary[key[0]]] - embeddings[dictionary[key[1]]])
            
    if (len(r)==0):
        print(keys, embeddings[dictionary[key[0]]])
    else:
        directionVec.append(sum(r)/len(r))
        
    r = []
    l = []
    for j in values:
        
        if (':' in j):
            value = j.split(":")
            r.append(embeddings[dictionary[value[0]]] -  embeddings[dictionary[value[1]]])
            l.append(j)  
    wordVec.append(r)
    words.append(l)


directionVec, wordVec = np.array(directionVec), np.array(wordVec)

result = []
for i in range(4):
    temp = []
    for j in range(len(directionVec)):
        temp.append(1-spatial.distance.cosine(directionVec[j,:].reshape(-1,1), wordVec[j,i,:].reshape(-1,1)))
    result.append(temp)
#    result.append(np.inner(directionVec, wordVec[:,i,:])/(np.linalg.norm(directionVec*np.linalg.norm(wordVec[:,i,:]))))


result = np.array(result)
result = result.T
print (result.shape)
words = np.array(words)
print (words.shape)
minIndexCol = np.argmin(result, axis=1)
maxIndexCol = np.argmax(result, axis=1)

opstr = ""
for i in range(len(words)):
    temp = ""
    temp +=  '\"' +('\" \"'.join(words[i])) + '\" ' 
    temp += '\"' +words[i][maxIndexCol[i]] + '" '        
    temp += '\"' + words[i][minIndexCol[i]] + '\"\n'
    opstr += temp

f = open(outputFile,'w')
f.write(opstr)
f.close()