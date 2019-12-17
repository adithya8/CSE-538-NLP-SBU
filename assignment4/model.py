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
        self.biDirection = layers.Bidirectional(tf.keras.layers.CuDNNGRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))
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

        op = self.biDirection(embed, mask = masked_embedding._keras_mask)

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

    def __init__(self, model_type: str, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = len(ID_TO_CLASS)
        self.model_type = model_type
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        if model_type == 'over':
            self.decoder = layers.Dense(units=self.num_classes)
            self.omegas1 = tf.Variable(tf.random.normal((hidden_size*2, hidden_size)))
            self.omegas2 = tf.Variable(tf.random.normal((hidden_size, (int)(hidden_size/2))))
            self.omegas3 = tf.Variable(tf.random.normal(((int)(hidden_size/2), 1)))
            self.biDirection = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))        
            self.batch_normed = layers.BatchNormalization(trainable=True) 
            

        elif model_type == 'sattn':
            self.biDirection = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))        
            self.decoder = layers.GRU(units=self.num_classes, return_sequences=True, activation='tanh')

        else:
            print ("CONV")
            self.decoder = layers.Dense(units=self.num_classes)
            self.biDirection = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, return_sequences=True, activation='tanh', recurrent_activation="tanh"))        
            self.conv = layers.Conv1D(filters = self.hidden_size, kernel_size = 3)
            self.omegas = tf.Variable(tf.random.normal((hidden_size, (int)(hidden_size/2))))

    
    def s_attention(self, inputs, scaled_=True):
        
        #Reference: https://github.com/ilivans/tf-rnn-attention
        attention = tf.matmul(inputs, inputs, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(inputs)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        attention = tf.nn.softmax(attention)  # [batch_size, sequence_length, sequence_length]
        return attention
    
    
    def attn(self, inter_outputs):

        if self.model_type == 'over':
            h_rnn_op = tf.nn.tanh(inter_outputs)
            alpha = tf.nn.relu(tf.linalg.matmul(h_rnn_op, self.omegas1))
            alpha2 = tf.nn.relu(tf.linalg.matmul(alpha, self.omegas2))
            alpha2 = self.batch_normed(alpha2)
            alpha3 = tf.nn.softmax(tf.linalg.matmul(alpha2, self.omegas3))
            output = tf.nn.tanh(tf.linalg.matmul(tf.transpose(alpha3, perm=[0,2,1]), inter_outputs))
        
        else:            
            h_rnn_op = tf.nn.tanh(inter_outputs)
            alpha = tf.nn.softmax(tf.linalg.matmul(h_rnn_op, self.omegas))
            output = tf.nn.tanh(tf.linalg.matmul(tf.transpose(alpha, perm=[0,2,1]), inter_outputs))
            
            
        return output


    def call(self, inputs, pos_inputs, training):
        
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        masking_layer = layers.Masking()
        unmasked_embedding = tf.cast(tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, 10]), tf.float32)        
        masked_embedding = masking_layer(unmasked_embedding)

        embed = tf.concat([word_embed, pos_embed], -1)
        
        if(training==True):
            embed = layers.Dropout(0.3)(embed)

        if self.model_type == 'over':

            op = self.biDirection(word_embed, mask = masked_embedding._keras_mask)

            if(training==True):
                op = layers.Dropout(0.3)(op)        

            attention = self.attn(op)

            if(training==True):
                attention = layers.Dropout(0.5)(attention)        
            
            logits = self.decoder(tf.reshape(attention, [-1 , 2*self.hidden_size]))
        
        else:
            op = self.biDirection(word_embed, mask = masked_embedding._keras_mask)            
            if(training==True):
                op = layers.Dropout(0.3)(op)        


            if self.model_type == 'sattn':
                attention = self.s_attention(op)
                key = self.decoder(op, mask = masked_embedding._keras_mask)
                
                def index1d(t):
                    return tf.reduce_min(tf.where(tf.equal(t, False)))

                indices = tf.map_fn(index1d, masked_embedding._keras_mask, dtype=tf.int64)
                condition = tf.math.greater(indices, 92233)
                case_true = tf.multiply(tf.ones([masked_embedding._keras_mask.shape[0]], tf.int64), masked_embedding._keras_mask.shape[-1])
                case_false = indices
                indices = tf.where(condition, case_true, case_false)
                indices = tf.math.subtract(indices, 1)

                key = layers.Dropout(0.3)(key)
                logits = tf.matmul(attention, key)

                logits_ = [logits[0, indices[0], :]]
                for i in range(1, tf.shape(embed)[0]):
                    logits_.append(logits[i, indices[i], :])
                logits_ = tf.stack(logits_)

                logits = logits_

            else:
                conv_op = self.conv(op)
#                if(training==True):
#                    conv_op = layers.Dropout(0.3)(conv_op)        

                attention = self.attn(conv_op)

#                if(training==True):
#                    attention = layers.Dropout(0.5)(attention)                    

                logits = self.decoder(tf.reshape(attention, [tf.shape(embed)[0] , -1]))
            #logits = self.decoder(tf.reshape(attention, [-1 , self.hidden_size]))
        
        return {'logits': logits} 
