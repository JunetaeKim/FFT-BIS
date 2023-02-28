import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K,losses
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Permute, Multiply, Flatten, Add, LSTM, Bidirectional
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
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
    
  
    Fs = 100  # Sampling frequency = 100 Hz
    DimSize = 30  # Length of EEG = 30 s
    SignalSize = 1  # Number of features 
    EEGmin = -1477.0
    EEGmax = 1800.0

    Half = int(Fs/2)
    InputWindowSize = Fs * DimSize
    DRrate = 0.5

    def SigProcessing (vector):
        Casted = tf.cast(vector, tf.complex64) 
        fft = tf.signal.fft(Casted)
        Scaled_fft =fft[:,:,:,:Half]
        Abs_fft = tf.abs(Scaled_fft)*(1/(Half*2.5)) # Relative amplitude
        Phase_ang = tf.math.angle(Scaled_fft)*180/np.pi + K.epsilon()
        return [Abs_fft,Phase_ang]

    def TimeAxis(input_batch):
        np_constant = [np.arange(1,DimSize+1,1).reshape(1, DimSize) for i in range(SignalSize)]
        tf_constant = K.constant(np.concatenate(np_constant,axis=0))
        batch_size = K.shape(input_batch)[0]
        constantVal = tf.tile(tf.expand_dims(tf_constant,0), ([batch_size,1,1]))
        return constantVal


    # Input : 30 s of EEG sampled at 100 Hz
    InputVec = Input(shape=(InputWindowSize, SignalSize), name='Input')
    PermutedDense1 = Permute((2,1), name='Permute')(InputVec)
    InputReStructure =Lambda(lambda x:K.reshape(x,(-1, SignalSize, DimSize, Fs)))(PermutedDense1) 

    # 1. FFT layers (Fig. 3)
  
    Abs_fft = Lambda(SigProcessing)(InputReStructure)[0]
    Phase_ang = Lambda(SigProcessing)(InputReStructure)[1]

    Sig1_Amp_inp = Lambda(lambda x:x[:, 0,:,:])(Abs_fft)
    Sig1_Phase_inp = Lambda(lambda x:x[:, 0,:,:])(Phase_ang)

    # 2. LSTM layers (Fig. 4)
    
    Sig1_Amp = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig1_Amp_inp, training=True)
    Sig1_Phase = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig1_Phase_inp, training=True)

    Sig_concat = Concatenate()([Sig1_Amp,Sig1_Phase])
    Sig_concat = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)
    Sig_concat = Bidirectional(LSTM(5, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)

    Amp_out = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)
    Phase_out = Bidirectional(LSTM(10, dropout=DRrate, activation='softsign', return_sequences=True))(Sig_concat, training=True)

    # 3. Interpretable layers (Fig. 5)

    Amp_weibul = Dense(10, activation='relu')(Amp_out)
    Amp_weibul = Dense(2, activation='softplus')(Amp_weibul)
    Amp_Alpha = Lambda(lambda x:x[:,0:1,0:1])(Amp_weibul)
    Amp_Beta = Lambda(lambda x:x[:,0:1,1:2])(Amp_weibul)
    TimeAxisVec = (Lambda(TimeAxis)(Amp_weibul))
    Amp_Time_Effect = 1-tf.exp(-(TimeAxisVec/Amp_Alpha)**Amp_Beta) # amplitude time-decayed effect
    Amp_out = Concatenate()([Amp_out, Amp_weibul])
    Amp_att = Dense(25, activation='elu')(Amp_out)
    Amp_att = Dense(Half, activation='softmax')(Amp_att) # amplitude attention

    Phase_weibul = Dense(10, activation='relu')(Phase_out)
    Phase_weibul = Dense(2, activation='softplus')(Phase_weibul)
    Phase_Alpha = Lambda(lambda x:x[:,0:1,0:1])(Phase_weibul)
    Phase_Beta = Lambda(lambda x:x[:,0:1,1:2])(Phase_weibul)
    TimeAxisVec = (Lambda(TimeAxis)(Phase_weibul))
    Phase_Time_Effect =1-tf.exp(-(TimeAxisVec/Phase_Alpha)**Phase_Beta) # phase time-decayed effect
    Phase_out = Concatenate()([Phase_out, Phase_weibul])
    Phase_att = Dense(25, activation='elu')(Phase_out)
    Phase_att = Dense(Half, activation='softmax')(Phase_att) # phase attention

    # 4. BIS calculation layers (Fig. 6) 
    
    Amp_end = Multiply(name='Amp_end')([Sig1_Amp_inp, Permute((2,1))(Amp_Time_Effect), Amp_att])
    Amp_end = Lambda(lambda x:tf.reduce_sum(x, axis=(1,2), keepdims=True)+K.epsilon())(Amp_end) # amplitude context
    Phase_end = Multiply(name='Phase_end')([Sig1_Phase_inp,Permute((2,1))(Phase_Time_Effect), Phase_att])
    Phase_end = Lambda(lambda x:tf.reduce_sum(x, axis=(1,2), keepdims=True)+K.epsilon())(Phase_end) # phase context

    Out = Add()([Amp_end,Phase_end])
    Out = Flatten(name='Add_end')(Out)

    model = Model(InputVec, Out)
          
    
    lrate = 0.001
    decay = 1e-7
    adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
    rmsprop = keras.optimizers.RMSprop(learning_rate=lrate, rho=0.9)

    model.compile(loss= losses.mean_squared_error, optimizer='adam') #losses.mean_squared_error
    
    SaveFilePath = './Logs/ModelTrain_{epoch:d}_{loss:.7f}_{val_loss:.7f}.hdf5'
    checkpoint = ModelCheckpoint(SaveFilePath,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto')
    earlystopper = EarlyStopping(monitor='val_loss', patience=2000, verbose=1)
    history = LossHistory()
    model.fit(trEEG[:], trBIS[:], validation_data = (valEEG[:],valBIS[:]) ,shuffle=True, batch_size=12000, epochs=10000, verbose=1, callbacks=[history,earlystopper,checkpoint])
    
    LossRes = np.concatenate([np.reshape(history.losses, (-1,1)), np.reshape(history.val_losses, (-1,1))])
    np.save('./Logs/ModelTrain_RE2LossRes.npy', LossRes)
    
