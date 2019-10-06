import os, sys
import pickle
import numpy as np
import pdb
from scipy import spatial



model_path = './models/'
if(len(sys.argv)>1):
    if sys.argv[1] == 'nce':
        loss_model = 'nce'
    else:
        loss_model = 'cross_entropy'

outputFile = './word_analogy_dev_sample_predictions.txt'
if(len(sys.argv)>2):
    outputFile = str(sys.argv[2])




model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
f = open('./word_analogy_dev.txt','r')
a = f.readlines()
f.close()

c = []
for i in a:
    b = []
    temp = i.strip().split('"')
    for j in temp:
        if (':' in j):
            b.append(j.split(':'))
    c.append(b)

d = []
for i in c:
    e = []
    for j in i:
        e.append([ embeddings[dictionary[j[0]]], embeddings[dictionary[j[1]]] ]) 
    d.append(e)


# d is going to be (#examples x #instPerExample x pairs x embeddings) 
# Out of 7 instances, the first 3 are the ones to learn from, the last four are the ones to find similarity for

d = np.array(d)
#pdb.set_trace()
print (d.shape)


directionVec = (d[:,0:3,0,:] - d[:,0:3,1,:])
directionVec = np.mean(directionVec, axis=1)
wordVec = (d[:,3:,0,:] - d[:,3:,1,:])
print (wordVec.shape)


# Cosine similarity: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists 
result = []
for i in range(4):
    temp = []
    for j in range(len(directionVec)):
        temp.append(1-spatial.distance.cosine(directionVec[j,:].reshape(-1,1), wordVec[j,i,:].reshape(-1,1)))
    result.append(temp)
#    result.append(np.inner(directionVec, wordVec[:,i,:])/(np.linalg.norm(directionVec*np.linalg.norm(wordVec[:,i,:]))))


result = np.array(result)
result = result.T
pdb.set_trace()
print (result.shape)
minIndexCol = np.argmin(result, axis=1)
maxIndexCol = np.argmax(result, axis=1)


opstr = ""
for i in range(len(c)):
    temp = ""
    for j in range(4):
        temp += '"' + ":".join(c[i][3:][j]) + '" '
    temp += '"' + ":".join(c[i][3:][minIndexCol[i]]) + '" '
    temp += '"' + ":".join(c[i][3:][maxIndexCol[i]]) + '"\n'
    
    opstr += temp

f = open(outputFile,'w')
f.write(opstr)
f.close()

