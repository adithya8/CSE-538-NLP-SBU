# Readme for Assignment 1 - CSE 538: fall 2019

## Task 1: Batch generation

- Before I proceeded with the batch generation I tried to gather a basic understanding of what the basic variables meant.
- I set the data_index to the value of skip_window if it is pointing to 0, since it needs to take 'skip_window' number of words to the left.
- I then reevaluate data_index to data_index%len(data) so to avoid it going out of index.
- For each function call, I repeat the batch generation process until the natch_size is met as described below by taking one data_index at a time and generate the label and batch pairs.
- First I straight away store the data at data_index for 'num_skips' times in labels since, 'num_skips' is the number of pairs to be formed.
- Then I gather all the words that are in 'skip_window' distance on either sides of the data_index.
- I randomly sample 'num_skip' words from the aforementioned list of words and I store it in batch.
- I increment the current batch size by 'num_skips' and move the data_index to the next word in data.
  

## Task 2: Cross Entropy

- The implementation of cross entropy is pretty much straight forward. 
- There are two terms to cross entropy function.
- The first term is the vector product of each of the context word and target word pairs' embeddings.
- I just made sure to arrive at a dimension of [batchsize x 1].
- The second term took some time for me to understand in terms of implementation.
- I set out by arriving at [batch_size x 1] as the output dimension.
- The second term involved the the product of each of context word embeddings to each of the target word embeddings.
- Appropriate dimensions were reduced by summing across an axis to arrive at [batch_size x 1].

## Task 3: NCE

- The Noise Contrastive Estimation required a bit more of work, since it requied me to understand the implementation of the paper. 
- I went with the same approach of arriving at [batch_size x 1].
- Hence I broke down the equation into two parts and two sub parts in each part.
- The first part of the NCE equation was similar to the cross entropy's numerator with an additional log of unigram probabilities.
- The second part involved modelling the negative words 'k' times per context word pairs and hence batch_size x k times for the entire batch.
- Appropriate transposes and reduction by sum across an axis was implemented to arrive at a dimension [batch_size x 1]

## Task 4: Word Analogy

- For each of the model, the embeddings was extracted for each word pair in the dev dataset and was represented as (914,7, 2,128).
- The first 3 word pairs in the second axis was taken and then the vectorized difference was computed and averaged to get the average of 'direction vector'.
- Difference between the word pairs for each of the following 4 word pairs were computed and the cosine similarity was computed against the 'direction vector' 
- The one with the least value was taken to be the most relevant word pair and the one with highest value was taken to be least relevant word pair.
  
## Task 5: Top 20

- For each of the words in {american, first , would}, the embeddings were retrieved from a partcular model.
- The cosine similarity between these words and the entire vocabulary was computed and sorted in increasing order.
- The first twenty similarity values were reported.

## Hyperparameters

### Cross Entropy

hyperParams = {
    'loss_model': "cross_entropy",
    'batch_size': 128,
    'embedding_size': 128,
    'skip_window': 2,
    'num_skips': 4,
    'num_sampled': None,
    'max_num_steps': 50001,
    'vocabulary_size': 100000
}

[Cross entropy model](https://drive.google.com/file/d/1IVd_s5rgcJViaSPbyHj_MEHSM10-aGW5/view?usp=sharing)

### NCE

hyperParams = {
    'loss_model': "NCE",
    'batch_size': 128,
    'embedding_size': 128,
    'skip_window': 4,
    'num_skips': 8,
    'num_sampled': 64,
    'max_num_steps': 100001,
    'vocabulary_size': 100000
}

[NCE model](https://drive.google.com/file/d/18Qm3AJDywxmPbCluuX8xpz_GDXYtV_CI/view?usp=sharing)

### Models folder

[folder](https://drive.google.com/drive/folders/1qjk2FgYKtt42XnhV_SImAM5K33dJvHEs?usp=sharing)