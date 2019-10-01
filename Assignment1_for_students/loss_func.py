import tensorflow as tf

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

    A = tf.reduce_sum(tf.multiply(inputs, true_w), axis=1)
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, true_w, transpose_a=True)), axis=1))

    return tf.subtract(B, A)

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

        

    A = tf.log(tf.sigmoid())
    B = tf.reduce_sum(tf.log(tf.sigmoid()),axis = 1)

    return tf.add(A,B)
