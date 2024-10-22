# inbuilt lib imports:
from typing import Dict
import math
import pdb

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        t = tf.fill(tf.shape(vector), 3.0)
        #print ("t.shape: ",t.shape)
        #print ("Pow shape: ", tf.pow(vector, t).shape)
        return tf.pow(vector, t)
        raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embeddings = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim], stddev=1/math.sqrt(vocab_size)))
        

        self.weight = {
            'hidden': tf.Variable(tf.random.truncated_normal([num_tokens*embedding_dim, hidden_dim], stddev=1/math.sqrt(num_tokens*embedding_dim))),
            'output': tf.Variable(tf.random.truncated_normal([hidden_dim, num_transitions], stddev=1/math.sqrt(hidden_dim)))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random.truncated_normal([hidden_dim], stddev=1/math.sqrt(hidden_dim))),
            #'output': tf.variable(tf.random.normal([num_transitions, embedding_dim]))
        }

        self.regLambda = regularization_lambda
        
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        x = tf.nn.embedding_lookup(self.embeddings, inputs)
        x = tf.reshape(x, [inputs.shape[0],-1])
        
        hidden_layer = tf.add(tf.linalg.matmul(x, self.weight['hidden']), self.biases['hidden'])
        
        hidden_layer = self._activation(hidden_layer)
        
        logits = tf.linalg.matmul(hidden_layer, self.weight['output'])
        
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        #Calculating Cross entropy: https://stackoverflow.com/questions/36078411/tensorflow-are-my-logits-in-the-right-format-for-cross-entropy-function/36086477#36086477
        condition = tf.equal(labels, -1)
        case_true = tf.constant(0, shape=labels.shape, dtype=tf.float32)
        case_false = tf.constant(1, shape=labels.shape, dtype=tf.float32)
        labels_mask = tf.where(condition, case_true, case_false)

        labels = tf.dtypes.cast(labels,tf.float32)
        actual_labels = tf.math.multiply(labels, labels_mask)
        
        logits_sm = tf.math.exp(tf.math.multiply(logits,labels_mask))
        logits_sm = tf.math.divide(logits_sm,tf.reduce_sum(logits_sm, axis=1, keepdims=1))

        loss = -tf.reduce_mean(tf.math.reduce_sum(actual_labels * tf.math.log(logits_sm + 10e-12), axis=1, keepdims=True))

        regularization = (tf.nn.l2_loss(self.weight['hidden']) + tf.nn.l2_loss(self.weight['output']))
        regularization = tf.fill(tf.shape(regularization),self.regLambda/2.0)*regularization
        # TODO(Students) End
        return loss + regularization
