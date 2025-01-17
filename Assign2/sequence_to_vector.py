# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models 


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.nnLayer = []
        self.dropout = dropout
        #self.denseLayer.append((tf.math.reduce_mean(self.x, axis=1)))
        # Send the input directly
        for i in range(num_layers):
            self.nnLayer.append(layers.Dense(units = self._input_dim, activation='relu'))
            # Include dropout
        ## TO DO: combined_vector stacking and layer_representation
        # ...
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
       
        #mask = [sequence_mask for i in range(self._input_dim)]
        #mask = tf.stack(mask, axis=2)
        
        if(training==True):
            drop_mask = tf.random.uniform(vector_sequence.shape[:2])
            #reference : https://stats.stackexchange.com/questions/240338/given-bernoulli-probability-how-to-draw-a-bernoulli-from-a-uniform-distribution
            comp_op = tf.greater(drop_mask, self.dropout) # returns boolean tensor
            drop_mask = tf.cast(comp_op, tf.float32) # casts boolean tensor into float32
            sequence_mask = tf.multiply(sequence_mask, drop_mask)

        new_mask = []
        for i in range(sequence_mask.shape[0]):
            if(tf.math.reduce_sum(sequence_mask[i]) == 0):
                new_mask.append(tf.cast(tf.ones(sequence_mask.shape[1]), tf.float32))
            else:
                new_mask.append(sequence_mask[i])

        mask = [new_mask for i in range(self._input_dim)]
        mask = tf.stack(mask, axis=2)

        # Num words in each sentence
        num_words_after_mask = tf.math.reduce_sum(mask, axis=1)

        input_vec = tf.multiply(vector_sequence,mask)
        # To only divide by the number of words after dropout for each sentence
        layer_op = tf.math.divide(tf.math.reduce_sum(input_vec, axis=1), num_words_after_mask)
        
        self.output_layer = []
        for layer in self.nnLayer:
            layer_op = layer(layer_op)
            self.output_layer.append(layer_op)
        
        combined_vector = layer_op
        layer_representations = tf.transpose(self.output_layer, perm=[1,0,2])
    
        # ...
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.nnLayer = []

        for i in range(num_layers):
            self.nnLayer.append(tf.keras.layers.GRU(self._input_dim, return_sequences=True))
        # ...
        # TODO(students): end


    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        #mask = [sequence_mask for i in range(self._input_dim)]
        #mask = tf.stack(mask, axis=2)
        
        #layer_op = tf.multiply(vector_sequence,mask)
        layer_op=vector_sequence
        self.output_layer = []
        i = 0 
        for layer in self.nnLayer:
            layer_op = layer(layer_op, mask=sequence_mask)
            self.output_layer.append(layer_op[:,-1,:])
        
        combined_vector = layer_op[:,-1,:]
        layer_representations = tf.transpose(self.output_layer, perm=[1,0,2])

        # ...
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
