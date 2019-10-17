import os
import word2vec_basic1
from pprint import pprint
import pdb
from datetime import datetime
import json

hyperParams = {
    'loss_model': "",
    'batch_size': "",
    'embedding_size': 128,
    'skip_window': "",
    'num_skips': "",
    'num_sampled': "",
    'max_num_steps': "",
    'run_word_analogy': "Y",
    'run_perl': "Y",
    'vocabulary_size': 100000
}

loss_models = ['cross_entropy', 'nce']

f = open("progress_so_far.txt", "a")
f.write(str(datetime.now())+"\n")
f.close()

def effect_process(hyperParams):
    f = open("progress_so_far.txt", "a")
    f.write(json.dumps(hyperParams))
    f.write("\n")
    f.close()
    word2vec_basic1.word2vec(hyperParams)


for max_num_steps in [50001,100001,200001,250001]:
    hyperParams['max_num_steps'] = max_num_steps
    for skip_window in [2, 3]:
        hyperParams['skip_window'] = skip_window
        for num_skips in range(2*skip_window-2, 2*skip_window+1):
            hyperParams['num_skips'] = num_skips
            for loss_model in loss_models:
                hyperParams['loss_model'] = loss_model
                if(loss_model == 'nce'):
                    hyperParams['batch_size'] = 128
                    for num_sampled in [16,32,64,128]:
                        hyperParams['num_sampled'] = num_sampled
                        effect_process(hyperParams)
                else:
                    for batch_size in [64, 128]:
                        hyperParams['batch_size'] = batch_size
                        effect_process(hyperParams)
                    

                        

                    


'''
for max_num_steps in [50001]:
    hyperParams['max_num_steps'] = max_num_steps
    for batch_size in [128]:
        hyperParams['batch_size'] = batch_size
        for skip_window in [3]:
            hyperParams['skip_window'] = skip_window
            for num_skips in range(2*skip_window-2, 2*skip_window-1):
                hyperParams['num_skips'] = num_skips
                hyperParams['loss_model'] = 'nce'
                if(hyperParams['loss_model'] == 'nce'):
                    for num_sampled in [16,32]:
                        hyperParams['num_sampled'] = num_sampled
                    pprint (hyperParams)
                    word2vec_basic1.word2vec(hyperParams)
                            

'''