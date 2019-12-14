import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
import pdb
from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.biDirection = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        h_rnn_op = tf.nn.tanh(rnn_outputs)
        alpha = tf.nn.softmax(tf.linalg.matmul(h_rnn_op, self.omegas))
        #print ("Alpha: ",alpha.shape)
        #print ("Alpha Transpose: ",  tf.transpose(alpha, perm=[0,2,1]).shape)
        output = tf.nn.tanh(tf.linalg.matmul(tf.transpose(alpha, perm=[0,2,1]), rnn_outputs))
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        #print ("word_embed shape: ", word_embed.shape)
        masking_layer = layers.Masking()
        unmasked_embedding = tf.cast(tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, 10]), tf.float32)        
        masked_embedding = masking_layer(unmasked_embedding)
        
        embed = tf.concat([word_embed, pos_embed], -1)

        if(training==True):
            embed = layers.Dropout(0.3)(embed)

        op = self.biDirection(word_embed, mask = masked_embedding._keras_mask)

        if(training==True):
            op = layers.Dropout(0.3)(op)
        #print (op.shape)
        
        attention = self.attn(op)
        
        if(training==True):
            attention = layers.Dropout(0.5)(attention)        

        logits = self.decoder(tf.reshape(attention, [-1 , 2*self.hidden_size]))
        #print (output.shape)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas1 = tf.Variable(tf.random.normal((hidden_size, (int)(hidden_size/2))))
        self.omegas2 = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.conv = Conv1D(filters = hidden_size, kernel_size = 3, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, use_bias=True, kernel_initializer='glorot_uniform')
        #self.mpool = MaxPooling1D(pool_size=3, strides=1)
        self.biDirection = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))        
        ### TODO(Students END

    def attn(self, inter_outputs):
        ### TODO(Students) START
        # ...
        h_rnn_op = tf.nn.tanh(inter_outputs)
        if(inter_outputs.shape[-1] == self.hidden_size):
            alpha = tf.nn.softmax(tf.linalg.matmul(h_rnn_op, self.omegas1))
        else:
            alpha = tf.nn.softmax(tf.linalg.matmul(h_rnn_op, self.omegas2))
        #print ("Alpha: ",alpha.shape)
        #print ("Alpha Transpose: ",  tf.transpose(alpha, perm=[0,2,1]).shape)
        output = tf.nn.tanh(tf.linalg.matmul(tf.transpose(alpha, perm=[0,2,1]), inter_outputs))
        ### TODO(Students) END

        return output


    def call(self, inputs, pos_inputs, training):
        
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        embed = tf.concat([word_embed, pos_embed], -1)
        #print ("embed: ",embed.shape)

        if(training==True):
            embed = layers.Dropout(0.3)(embed)

        convop = self.conv(embed)
        #print ("convop: ",convop.shape)

        #mpoolop = convop
        #mpoolop = self.mpool(convop)
        #print ("mpoolop: ",mpoolop.shape)

        interop = self.attn(convop)
        #print ("interop: ",interop.shape)

        bidirecop = self.biDirection(interop)        
        #print ("bidirecop: ",bidirecop.shape)

        attention = self.attn(bidirecop)
        #print ("attention: ",attention.shape)

        logits = self.decoder(tf.reshape(attention, [-1 , 2*self.hidden_size]))        

        ### TODO(Students END
        return {'logits': logits} 
