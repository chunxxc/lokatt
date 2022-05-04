import numpy as np
import tensorflow as tf
#tfa.register_all()
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import get as get_activation
from tensorflow.keras.layers import SimpleRNNCell, RNN, Layer
class SimpleRNNCellWithLayerNorm(SimpleRNNCell):
    def __init__(self, units, **kwargs):
        self.activation = get_activation(kwargs.get("activation", "tanh"))
        kwargs["activation"] = None
        super().__init__(units, **kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self, inputs, states):
        outputs, new_states = super().call(inputs, states)
        norm_out = self.activation(self.layer_norm(outputs))
        return norm_out, [norm_out]
'''
class Model_RESBi(tf.keras.Model):
  def __init__(self,kmer=5):
    super(Model_RESBi, self).__init__()
    self.initializer = tf.keras.initializers.HeNormal()
    self.output_dim = 4**kmer
    self.res1 = Residual_block(initializer=self.initializer)
    self.res2 = Residual_block(initializer=self.initializer)
    #self.resx = Residual_block(dilat=[5,5,5],initializer=self.initializer)
    #self.res3 = Residual_block()
    #self.res4 = Residual_block()
    #self.res5 = Residual_block()
    #self.res6 = Residual_block()
    self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    self.bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    #self.biconvlstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.ConvLSTM1D(256,15,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    #self.biconvlstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.ConvLSTM1D(256,15,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    #self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    #self.bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1024,return_sequences=True,return_state=True,kernel_initializer=self.initializer))
    #lnLSTM1 = tfa.rnn.LayerNormLSTMCell(256,kernel_initializer=self.initializer)
    #lnLSTM2 = tfa.rnn.LayerNormLSTMCell(1024,kernel_initializer=self.initializer)
    #lnLSTM1 = tfa.rnn.LayerNormSimpleRNNCell(256)
    #lnLSTM2 = tfa.rnn.LayerNormSimpleRNNCell(1024)
    #lnLSTM1 = SimpleRNNCellWithLayerNorm(256)
    #lnLSTM2 = SimpleRNNCellWithLayerNorm(1024)
    #self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(lnLSTM1,return_sequences=True,return_state=True))
    #self.bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(lnLSTM2,return_sequences=True,return_state=True))
    #self.bilstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_k[2],return_sequences=True,return_state=True))
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    #self.embedding = tf.keras.layers.Embedding(self.output_dim,kmer2vec_embedding.shape[1],embeddings_initializer=keras.initializers.Constant(kmer2vec_embedding),trainable=False,)
    self.dense = tf.keras.layers.Dense(self.output_dim,activation=None,kernel_initializer=self.initializer)
  def call(self,x,training=0,temperature=1):
    x = self.res1(x,training)
    x = self.res2(x,training)
    #x = self.resx(x,training)
    x = self.dropout1(x, training=training)
    x,_,_,_,_ = self.bilstm1(x)
    x,_,_,_,_ = self.bilstm2(x)
    #h6,_,_,_,_ = self.bilstm3(h5)
    x = self.dropout2(x, training=training)
    x = self.dense(x)
    if not training:
      output = tf.nn.softmax(x/temperature)
    else:
      output = tf.nn.softmax(x)
    return output  
######################################################################################
class Residual_block(tf.keras.layers.Layer):
  def __init__(self,channel=[32,64,32],kernel=[5,5,5],dilat=[1,1,1],initializer='glorot_uniform'):
    super(Residual_block, self).__init__()
    self.conv1 = tf.keras.layers.Conv1D(channel[0],kernel[0],activation=None,padding='same',use_bias=False,dilation_rate=dilat[0],kernel_initializer=initializer)
    self.conv2 = tf.keras.layers.Conv1D(channel[1],kernel[1],activation=None,padding='same',use_bias=False,dilation_rate=dilat[1],kernel_initializer=initializer)
    self.conv3 = tf.keras.layers.Conv1D(channel[2],kernel[2],activation=None,padding='same',use_bias=False,dilation_rate=dilat[2],kernel_initializer=initializer)
    self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
    self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
    self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
  def call(self,x,training=None):
    hx = self.conv1(x)
    #hx = self.bn1(hx,training=training)
    hx = self.ln1(hx)
    hx = tf.keras.activations.swish(hx)
    #hx = tf.nn.relu(hx)
    hx = self.conv2(hx)
    hx = self.ln2(hx)
    #hx = self.bn2(hx,training=training)
    hx = tf.keras.activations.swish(hx)
    #hx = tf.nn.relu(hx)
    hx = self.conv3(hx)
    hx = self.ln3(hx)
    #hx = self.bn3(hx,training=training)
    output = tf.keras.activations.swish(hx+x)
    #output = tf.nn.relu(hx+x)
    return output
######################################################################################
if __name__ == '__main__':
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.set_visible_devices(physical_devices[2], 'GPU')
  import pdb;pdb.set_trace()
  mymodel = Model_RESBi()
  test = mymodel(np.random.rand(1,4096,1).astype(np.float32))
