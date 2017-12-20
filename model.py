
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


def build_inputs(batch_size, num_steps):
    
    #===========================================================================================
    #Function to build tensor inputs to tensorflow
    #---------------------------------------------
    #batch_size: Number of samples per batch
    #num_steps: Number of sequences per sample
    #===========================================================================================
    
    #Declare the placeholders
    inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 74), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps, 3), name='labels')
    
    #Dropout layer probabilities
    keep_prob = tf.placeholder(tf.float32 , name='keep_prob')
        
    return inputs, targets, keep_prob


# In[3]:


def build_network(gru_size, num_layers, batch_size, keep_prob):
    
    #===========================================================================================
    #Function to build network. The network will be the following
    #LSTM > Dropout > LSTM > Dropout > Dense > Softmax
    #---------------------------------------------
    #lstm_size: Number of samples per batch
    #num_layers: Number of layers for LSTM
    #batch_size: Number of sequences per batch
    #keep_prob: Dropout probability for the 2 dropout layers
    #dense_hidden_units: Number of hidden units within the dense layer
    #===========================================================================================
    
    def build_cell(size, keep_prob):
        
        #Helper function to create an LSTM cell, just to make things a bit neat
        gru = tf.contrib.rnn.GRUCell(size)
        
        #Add drop out
        drop = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=keep_prob)
        
        return drop
    
    #Stack up LSTM layers, and then add a dense layer
    stacked_gru = tf.contrib.rnn.MultiRNNCell([build_cell(gru_size, keep_prob) for _ in range(0, num_layers)])
        
    #Initial State
    initial_state = stacked_gru.zero_state(batch_size, tf.float32)
    
    return stacked_gru, initial_state


# In[4]:


def build_output(stacked_gru, dense_hidden_units, in_size, n_class=3):
    
    #===========================================================================================
    # Function to build output function
    #---------------------------------------------
    # stacked_lstm: input tensor from stacked LSTM
    # dense_hidden_units: number of hidden units from dense layer
    # n_class: number of class probabilities
    #===========================================================================================
    
    #We have the dense layer, build logits
    seq_output = tf.concat(stacked_gru, axis=1)
    #print('seq_output:', seq_output.get_shape())
    
    x = tf.reshape(seq_output, [-1, in_size])
    #print('x:', x.get_shape())
    
    dense = tf.layers.dense(inputs=x, units=dense_hidden_units, activation=tf.nn.relu)
    
    softmax_w = tf.Variable(tf.truncated_normal((dense_hidden_units, n_class), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(n_class))
    
    #Calculate logit
    logits = tf.matmul(dense, softmax_w) + softmax_b
    #print('logits:', logits.get_shape())
    #Output
    out = tf.nn.softmax(logits, name='predictions')
    #print('out:', out.get_shape())
    
    return out, logits
    


# In[5]:


def build_loss(logits, targets, n_class=3):
    
    #===========================================================================================
    # Function to build output function
    #---------------------------------------------
    # dense: input tensor from dense layer
    # n_class: number of class probabilities
    #===========================================================================================
    
    #print('build_loss logits:', logits.get_shape())
    #print('build_loss targets:', targets.get_shape())
    
    #Reshape targets
    y_reshaped =  tf.reshape(targets, logits.get_shape())
    #print('y_reshaped:', y_reshaped.get_shape())
    
    #Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss
    


# In[6]:


def build_optimizer(loss, learning_rate, grad_clip):
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


# In[8]:


#Define LSTM Parameters and structure
class DirectionLSTM:
    
    #============================================================================================
    #Class Constructor
    #-----------------
    #Construct GRU object to carry around
    #
    #
    #============================================================================================
    def __init__(self, n_class=3, batch_size=128, num_steps=10, gru_size=50, dense_size=50, num_layers=2,
                 learning_rate=0.001, grad_clip=2, online=False):
        
        #If Online learning, we will be having batch_size of 1 and 1 time step
        if online == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        
        #Reset graph
        tf.reset_default_graph()
        
        #Build inputs
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        
        #Build LSTM Cell
        cell, self.initial_state = build_network(gru_size, num_layers, batch_size, self.keep_prob)
        
        #Link Up the RNN cells
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state
        
        #Get softmax predictions
        self.prediction, self.logits = build_output(outputs, dense_size, gru_size, n_class)
        
        #Loss and optimizer
        self.loss = build_loss(self.logits, self.targets, n_class)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        
        #print(self.targets.get_shape())
        #print(self.logits.get_shape())
        
        #Evaluation metrics
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(tf.reshape(self.targets, self.prediction.get_shape()), 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        


# In[ ]:




