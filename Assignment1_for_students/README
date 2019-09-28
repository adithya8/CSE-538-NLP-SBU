==========================================================================================

         Introduction to Natural Laguage Processing Assignment 1
 
==========================================================================================

1. In this assignment, you will be asked to:

  - generate batch for skip-gram model (word2vec_basic.py)
  - implement two loss functions to train word embeddings (loss_func.py)
  - tune the parameters for word embeddings 
  - apply learned word embeddings to word analogy task (word_analogy.py)


2. Generating batch
  You will generate small subset of training data, which is called batch.

  For skip-gram model, you will slide a window
  and sample training instances from the data insdie the window.

  [Example]
  Suppose that we have a text: "The quick brown fox jumps over the lazy dog."
  And batch_size = 8, window_size = 3

  "[The quick brown] fox jumps over the lazy dog"

  Context word would be 'quick' and predicting words are 'The' and 'brown'.
  This will generate training examples:
       context(x), predicted_word(y)
          (quick    ,       The)
          (quick    ,     brown)

  And then move the sliding window.
  "The [quick brown fox] jumps over the lazy dog"
  In the same way, we have two more examples:
      (brown, quick)
      (brown, fox)

  Moving the window again:
  "The quick [brown fox jumps] over the lazy dog"
  We get,
      (fox, brown)
      (fox, jumps)

  Finally we get two more instances from the moved window,
  "The quick brown [fox jumps over] the lazy dog"
      (jumps, fox)
      (jumps, over)

  Since now we have 8 training instances, which is the batch size,
  stop generating this batch and return batch data.

  data_index is the index of a word. You can access a word using data[data_index].
  batch_size is the number of instances in one batch.
  num_skips is the number of samples you want to draw in a window(in example, it was 2).
  skip_windows decides how many words to consider left and right from a context word(so, skip_windows*2+1 = window_size).
  batch will contains word ids for context words. Dimension is [batch_size].
  labels will contains word ids for predicting words. Dimension is [batch_size, 1].


3. Analogies using word vectors

  You will use the word vectors you learned from both approaches(cross entropy and NCE) in the following word analogy task.

  Each question/task is in the following form. 
  -------------------------------------------------------------------------------------
  Consider the following word pairs that share the same relation, R:

      pilgrim:shrine, hunter:quarry, assassin:victim, climber:peak
  
  Among these word pairs,

  (1) pig:mud
  (2) politician:votes
  (3) dog:bone
  (4) bird:worm

  Q1. Which word pairs has the MOST illustrative(similar) example of the relation R?
  Q2. Which word pairs has the LEAST illustrative(similar) example of the relation R?
  -------------------------------------------------------------------------------------


  One simple method to answer those questions will be measuring the similarities of difference vectors.

  [Difference Vector]
  Recall that vectors are representing some direction in space. 
  If (a, b) and (c, d) pairs are analogous pairs then the transformation from a to b (i.e., some x vector when added to a gives b: a + x = b) 
  should be highly similar to the transformation from c to d (i.e., some y vector when added to c gives d: b + y = d). 
  In other words, the difference vector (b-a) should be similar to difference vector (d-c). 

  This difference vector can be thought to represent the relation between the two words. 
  
  
  * Due to the noisy annotation data, the expected accuracy is not high. 
  The pre-trained model scores 31.4% overall accuracy. Improving this score 1~3% would be your goal. 


4. This package contains several files:

For word2vec

  - word2vec_basic.py: 
    This file is the main script for training word2vec model.
    You should fill out the part that generates batch from training data.
    Your trained model will be saved as word2vec.model
    Usage:
      python word2vec_basic.py [cross_entropy | nce]


  - loss_func.py
    This file have two loss functions but they should be completed.
    1. cross_entropy_loss - Loss we used in class. See lecture notes on word representations for details.
    2. nce_loss - Noise contrastive estimation loss described in the assignment pdf.

  - pretrained/word2vec

For analogy task

  - word_analogy.py
    You will write a code in this file for evaluating relation between pairs of words -- called MaxDiff question.
    (https://en.wikipedia.org/wiki/MaxDiff)
    You will generate a file with your predictions following the format of "word_analogy_sample_predictions.txt"

  - score_maxdiff.pl
    a perl script to evaluate YOUR PREDICTIONS on development data
    Usage:
      ./score_maxdiff.pl word_analogy_mturk_answers.txt <your prediction file> <output file of result> 

  - word_analogy_dev.txt
    Data for development. 
    Each line of this file is divided into "examples" and "choices" by "||".
        [examples]||[choices]
    "Examples" and "choices" are delimited by a comma.
      ex) "tailor:suit","oracle:prophesy","baker:flour"

  - word_analogy_dev_sample_predictions.txt
    A sample prediction file. Pay attention to the format of this file. 
    Your prediction file should follow this to use "score_maxdiff.pl" script.
    Each row is in this format:
     
      <pair1> <pair2> <pair3> <pair4> <least_illustrative_pair> <most_illustrative_pair>

    The order of word pairs should match their original order found in "word_analogy_dev.txt".


  - word_analogy_dev_mturk_answers.txt
    This is the answers collected using Amazon mechanical turk for "word_analogy_dev.txt". 
    The answers in this file is used as the correct answer and used to evaluate your analogy predictions. (using "score_maxdiff.pl")
    For your information, the answers here are a little bit noisy.

  - word_analogy_test.txt
    Test data file. When you are done experiment with your model, you will generate predictions for this test data using your best models (NCE/cross entropy).




5. What you should do:

  1. Implement batch generation
  2. Implement Cross Entropy Loss 
  3. Implement NCE Loss
  4. Try different parameters to train your word2vec model
  5. Write a code to evaluate relations between word pairs
  6. Find the least illustrative pair and most illustrative pair among four pairs of words.


6. What you should submit:
  
  ** PAY ATTENTION TO THE FILENAMES!!! 
  I will use scripts to organize your codes/files. 

  Create a single zip file ( <SBUID>.zip ) contains the following files:
      1. word2vec_basic.py
      2. loss_func.py
      3. your best word2vec models for Cross Entroy loss and NCE loss
        - word2vec_cross_entropy.model
        - word2vec_nce.model
      4. word_analogy.py
      5. Prediction on data generated for word_analogy_test.txt
        - word_analogy_test_predictions_cross_entropy.txt
        - word_analogy_test_predictions_nce.txt
      6. README file with explanation of your implementation
      7. Report as detailed in the assignment pdf
        - report_<SBUID>.pdf



