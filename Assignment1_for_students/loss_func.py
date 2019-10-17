import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    # Matmul vs Multiply: https://stackoverflow.com/questions/47583501/tf-multiply-vs-tf-matmul-to-calculate-the-dot-product
    # Ask Matt: Understand the objective clearly and explain your thought using Niranjan's slide
    # multiply does an element wise multiplication 
    # axis 1 -> sum along row

    A = tf.reduce_sum(tf.multiply(true_w, inputs), axis=1)
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(true_w, inputs, transpose_b=True)), axis=1))

    return tf.subtract(B, A)

def process_negative_samples(weights, biases, labels, sample, diffK = {'status': False, 'k': 0 }):
    # When diffK is set to false, then all the words in the batch have the same k 'negative samples'
    if(diffK['status'] == False):
        #All words get same negative samples
        negEmbed = tf.nn.embedding_lookup(weights, sample)
        negBias = tf.nn.embedding_lookup(biases, sample)
        return negEmbed, negBias

    else:
        #Each word gets a different negative sample
        k = max(len(sample), diffK['k'])
        

def logneg(arg):
    return tf.log(tf.math.maximum(tf.keras.backend.epsilon(), arg))

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    
        A = - log{Pr(w_o/w_c)} = - log{sig{u_c^Tu_o + b_o - log{k*Pr(w_o)}}}
        B = - sum{log{1-Pr(w_x/w_c)}} = - sum{log{1 - sig{u_c^Tu_x + b_x - log{k*Pr(w_x)}}}}   

    """
    # Think about various ways of sampling k negative examples.  

    # processing the unified labels.     
    # Same set of k negative samples for all words in a batch (or)
    # Different k negative samples for each word in the batch
    unigram_prob = np.array(unigram_prob, dtype=np.float32)
    negEmbed, negBias = process_negative_samples(weights, biases, labels, sample, diffK={'status': False, 'k': 0})
    posEmbed = tf.nn.embedding_lookup(weights, labels)
    posBias = tf.nn.embedding_lookup(biases, labels)
    
    subArg1 = tf.add(tf.reduce_sum(tf.multiply(inputs, posEmbed), axis=1), posBias)
    unigram_probab = tf.nn.embedding_lookup(unigram_prob,labels)
    #randD = tf.random_uniform((tf.shape(labels)[0],1), minval=0, maxval=0.01, dtype=tf.float32)
    subArg1_= logneg((tf.cast(tf.multiply(np.shape(sample)[0]/1.0, unigram_probab), dtype=tf.float32))+tf.keras.backend.epsilon())
    
    arg1 =  subArg1 - subArg1_

    subArg2 = tf.matmul(inputs, negEmbed, transpose_b=True)+negBias

#    randD = tf.random_uniform((len(sample),1), minval=0, maxval=0.01, dtype=tf.float32)
    randD = np.random.random(len(sample))
    subArg2_ = tf.reshape((logneg(tf.cast(tf.multiply(np.shape(sample)[0]/1.0,unigram_prob[sample]), dtype=tf.float32)+tf.keras.backend.epsilon())), (1, -1))
    arg2 = tf.sigmoid(tf.subtract(subArg2, subArg2_))


    A = tf.reduce_sum(logneg(tf.sigmoid(arg1)), axis=1)

    B = tf.reduce_sum(logneg(1-arg2+tf.keras.backend.epsilon()), axis=1)
    print (A)
    print (B)    

    return -A-B
