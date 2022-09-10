from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython.display import Image
from scipy.stats import sem, t
from statsmodels.tsa.stattools import adfuller 
from matplotlib.colors import LogNorm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import sklearn.metrics as metrics

from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Dense,Average,Concatenate,Lambda, Permute, Activation, dot, BatchNormalization,Multiply, Concatenate,Flatten, concatenate, Add,Dot,Reshape,Conv2D, Conv1D, LSTM,TimeDistributed,Bidirectional
import tensorflow as tf
from tensorflow.keras.backend import  expand_dims

from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K,losses
from tensorflow.keras.activations import softmax



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


if __name__ == "__main__":
    
    
    # TensorFlow wizardry
    config = tf.compat.v1.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # Create a session with the above options specified.
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    
    trEEG = np.load('../ProcessedData/EEGsim_Train.npy')
    trBIS = np.load('../ProcessedData/BISsim_Train.npy')
    valEEG = np.load('../ProcessedData/EEGsim_Val.npy')
    valBIS = np.load('../ProcessedData/BISsim_Val.npy')
    trBIS = np.round(trBIS, 2) 
    valBIS = np.round(valBIS, 2)
    minEEG = trEEG.min()
    maxEEG = trEEG.max()
    trEEG = (trEEG - minEEG)/(maxEEG-minEEG)
    valEEG = (valEEG - minEEG)/(maxEEG-minEEG)
    
  
    Fs = 100      # Sampling frequency
    DimSize = 30   # End of time 몇 초
    SignalSize=1 # Feature 수
    MaxCensoring = 10 # Censoring을 얼마나 할것인지 defualt 10
    EEGmin = -1477.0
    EEGmax = 1800.0

    Half = int(Fs/2)
    InputWindowSize = Fs *DimSize
    DRrate = 0.5

    def SigProcessing (vector):
        Casted = tf.cast(vector, tf.complex64) 
        fft = tf.signal.fft(Casted)
        Scaled_fft =fft[:,:,:,:Half]
        #Filtered_fft = tf.cast(Scaled_fft, tf.float32)
        Abs_fft = tf.abs(Scaled_fft)*(1/(Half*2.5)) # 상대적 진폭 Relative amplitude
        Phase_ang = tf.math.angle(Scaled_fft)*180/np.pi + K.epsilon()

        return [Abs_fft,Phase_ang]

    def TimeAxis(input_batch):
        np_constant = [np.arange(1,DimSize+1,1).reshape(1, DimSize) for i in range(SignalSize)]
        tf_constant = K.constant(np.concatenate(np_constant,axis=0))
        #tf_constant = tf.tile(tf.expand_dims(tf_constant,1), ([1,Half,1]))
        batch_size = K.shape(input_batch)[0]
        constantVal = tf.tile(tf.expand_dims(tf_constant,0), ([batch_size,1,1]))
        return constantVal


    ## Input side
    InputVec = Input(shape=(InputWindowSize, SignalSize), name='Input')
    PermutedDense1 = Permute((2,1), name='Permute')(InputVec)
    InputReStructure =Lambda(lambda x:K.reshape(x,(-1, SignalSize, DimSize, Fs)))(PermutedDense1) #Batch, element(EEG1, EEG2.....), windowID, freq, 

    Abs_fft = Lambda(SigProcessing)(InputReStructure)[0]
    Phase_ang = Lambda(SigProcessing)(InputReStructure)[1]

    Sig1_FFT_inp = Lambda(lambda x:x[:, 0,:,:])(Abs_fft)
    Sig1_Phase_inp = Lambda(lambda x:x[:, 0,:,:])(Phase_ang)


    Sig1_FFT = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig1_FFT_inp, training=True)
    Sig1_Phase = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig1_Phase_inp, training=True)

    Sig_concat = Concatenate()([Sig1_FFT,Sig1_Phase])
    Sig_concat = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)
    Sig_concat = Bidirectional(LSTM(5, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)

    FFT_out = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)
    Phase_out = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)

    ## FFT Time decay
    FFT_weibul = Dense(10, activation='relu')(FFT_out)
    FFT_weibul = Dense(2, activation='softplus')(FFT_weibul)
    FFT_Alpha = Lambda(lambda x:x[:,0:1,0:1])(FFT_weibul)
    FFT_Beta = Lambda(lambda x:x[:,0:1,1:2])(FFT_weibul)
    TimeAxisVec = (Lambda(TimeAxis)(FFT_weibul))
    FFT_Time_Effect = 1-tf.exp(-(TimeAxisVec/FFT_Alpha)**FFT_Beta)


    ## FFT attention
    FFT_out = Concatenate()([FFT_out, FFT_weibul])
    FFT_att = Dense(25, activation='elu')(FFT_out)
    FFT_att = Dense(Half, activation='softmax')(FFT_att)


    ## FFT Context
    FFT_end = Multiply(name='FFT_end')([Sig1_FFT_inp, Permute((2,1))(FFT_Time_Effect), FFT_att])
    FFT_end = Lambda(lambda x:tf.reduce_sum(x, axis=(1,2), keepdims=True)+K.epsilon())(FFT_end)


    ## Phase Time decay
    Phase_weibul = Dense(10, activation='relu')(Phase_out)
    Phase_weibul = Dense(2, activation='softplus')(Phase_weibul)
    Phase_Alpha = Lambda(lambda x:x[:,0:1,0:1])(Phase_weibul)
    Phase_Beta = Lambda(lambda x:x[:,0:1,1:2])(Phase_weibul)
    TimeAxisVec = (Lambda(TimeAxis)(Phase_weibul))
    Phase_Time_Effect =1-tf.exp(-(TimeAxisVec/Phase_Alpha)**Phase_Beta)

    ## Phase attention
    Phase_out = Concatenate()([Phase_out, Phase_weibul])
    Phase_att = Dense(25, activation='elu')(Phase_out)
    Phase_att = Dense(Half, activation='softmax')(Phase_att)

    ## FFT Context
    Phase_end = Multiply(name='Phase_end')([Sig1_Phase_inp,Permute((2,1))(Phase_Time_Effect), Phase_att])
    Phase_end = Lambda(lambda x:tf.reduce_sum(x, axis=(1,2), keepdims=True)+K.epsilon())(Phase_end)

    Out = Add()([FFT_end,Phase_end])
    Out = Flatten(name='Add_end')(Out)

    model = Model(InputVec, Out)
    
    print(model.summary())
        
    
    lrate = 0.001
    decay = 1e-7
    adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
    rmsprop = keras.optimizers.RMSprop(learning_rate=lrate, rho=0.9)

    model.compile(loss= losses.mean_squared_error, optimizer='adam') #losses.mean_squared_error
    
    # MODEL_SAVE_FOLDER_PATH + '{epoch:d}-{val_loss:.6f}.hdf5'
    SaveFilePath = './Logs/ModelTrain_{epoch:d}_{loss:.7f}_{val_loss:.7f}.hdf5'
    checkpoint = ModelCheckpoint(SaveFilePath,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto')
    earlystopper = EarlyStopping(monitor='val_loss', patience=2000, verbose=1)
    history = LossHistory()
    model.fit(trEEG[:], trBIS[:], validation_data = (valEEG[:],valBIS[:]) ,shuffle=True, batch_size=12000, epochs=15000, verbose=1, callbacks=[history,earlystopper,checkpoint])
    
    LossRes = np.concatenate([np.reshape(history.losses, (-1,1)), np.reshape(history.val_losses, (-1,1))])
    np.save('./Logs/ModelTrain_RE2LossRes.npy', LossRes)
    