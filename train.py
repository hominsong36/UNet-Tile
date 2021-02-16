import numpy as np
import pickle
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau 
from sklearn.utils import shuffle

# Training parameters
batch_size = 16
epochs = 500

# Data and target (label) load
base_dir = 'dataset/'

X_train = np.load(base_dir + 'X_train.npy')
y_train = np.load(base_dir +'y_train.npy')
X_train_aug = np.load(base_dir +'X_train_aug.npy')
y_train_aug = np.load(base_dir +'y_train_aug.npy')
X_val = np.load(base_dir +'X_val.npy')
y_val = np.load(base_dir +'y_val.npy')

X_train = (X_train/255).astype('float32')
y_train = (y_train/255).astype('float32')
X_train_aug = (X_train_aug/255).astype('float32')
y_train_aug = (y_train_aug/255).astype('float32')
X_val = (X_val/255).astype('float32')
y_val = (y_val/255).astype('float32')

X_train = np.concatenate((X_train, X_train_aug), axis=0)
y_train = np.concatenate((y_train, y_train_aug), axis=0)

X_train, y_train = shuffle(X_train, y_train, random_state=17)

print('Train/Val data loaded')

#%%
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

# Learning rate schedule
def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 800:
        lr *= 1e-2
    elif epoch > 600:
        lr *= 5e-2
    elif epoch > 400:
        lr *= 1e-1
    elif epoch > 200:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr

  
# Define the dice loss
def dice_loss(y_true, y_pred):
     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3)) + 1e-6
     denominator = tf.reduce_sum(y_true, axis=(1,2,3)) + tf.reduce_sum(y_pred, axis=(1,2,3)) + 1e-6
     
     return 1 - numerator / denominator

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Build a U-Net model with residual connections
def resunet(pretrained_weights = None,input_size = (256,256,3)):
    
    def res_block(layer_input, n_filters, dropout_rate):

            rb_out = BatchNormalization()(layer_input)
            rb_out = Activation('relu')(rb_out)
            rb_out = Conv2D(n_filters, 3, padding = 'same', kernel_initializer = 'he_normal')(rb_out)
            rb_out = Dropout(dropout_rate)(rb_out)
            rb_out = BatchNormalization()(rb_out)
            rb_out = Activation('relu')(rb_out)
            rb_out = Conv2D(n_filters, 3, padding = 'same', kernel_initializer = 'he_normal')(rb_out)
            rb_out = Dropout(dropout_rate)(rb_out)
            rb_out = add([layer_input, rb_out])
    
            return rb_out
        
    inputs = Input(input_size)
    conv0 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    
    conv1 = res_block(conv0, 8, 0.5)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Conv2D(16, 1, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    
    conv2 = res_block(pool1, 16, 0.5)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Conv2D(32, 1, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    
    conv3 = res_block(pool2, 32, 0.5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    
    conv4 = res_block(pool3, 64, 0.5)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    
    conv5 = res_block(pool4, 128, 0.5)
    
    up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = add([conv4,up6])
    conv6 = res_block(merge6, 64, 0.5)
        
    up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = add([conv3,up7])
    conv7 = res_block(merge7, 32, 0.5)
    
    up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = add([conv2,up8])
    conv8 = res_block(merge8, 16, 0.5)
        
    up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = add([conv1,up9])
    conv9 = res_block(merge9, 8, 0.5)
      
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'linear', kernel_initializer = 'he_normal')(conv10)
    conv10 = add([conv0, conv10])
    conv10 = Activation('sigmoid')(conv10)  
    
    model = Model(input = inputs, output = conv10)
    
    model.compile(optimizer = Adam(lr = lr_schedule(0)), loss = dice_loss, metrics = [mean_iou])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
  
#%%
################  Train the model   #######################
###########################################################
save_dir = 'resunet_weights_histories/'

model = resunet()
print('ResUnet model has been built.')

model_checkpoint = ModelCheckpoint(save_dir + 'resunet_tile.hdf5', monitor='val_loss',verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode = 'min')
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.5e-6)

history = model.fit(X_train, y_train,
                    validation_data = (X_val, y_val),
                    epochs=epochs,
                    batch_size = batch_size,
                    shuffle = True,
                    callbacks = [
                                  early_stopping,
                                  model_checkpoint,
                                  lr_scheduler,
                                  lr_reducer
                                  
                    ]
                   )
histories = history.history

######### Saving histories #########
with open(save_dir + 'resunet_tile-history.pkl', 'wb') as f:
    pickle.dump(histories, f)