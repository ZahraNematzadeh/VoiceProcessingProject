import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def kfold_training(melspect_train_array, melspect_test_array,
                   y_train_one_hot, y_test_one_hot, test_count,
                   model, scheduler,checkpointer, k_fold,
                   Batch_size, num_epochs, boolean_leaf):
    j = 0
    num_k = k_fold
    BATCH_SIZE = Batch_size
    num_epochs = num_epochs
    Batch_size_test = test_count
    data_kfold = pd.DataFrame()
    model_history = []

    kfold = KFold(n_splits = num_k, shuffle=True, random_state=42)
    for train_idx, val_idx in list(kfold.split(melspect_train_array,y_train_one_hot)):
        print("Training on Fold: ",j+1)
        if boolean_leaf:
            train_idx_tensor = tf.convert_to_tensor(train_idx, dtype=tf.int32)
            val_idx_tensor = tf.convert_to_tensor(val_idx, dtype=tf.int32)
            x_train_df = tf.gather(melspect_train_array, train_idx_tensor)
            x_valid_df = tf.gather(melspect_train_array, val_idx_tensor)
            y_train_fold = tf.gather(y_train_one_hot, train_idx_tensor)
            y_val_fold = tf.gather(y_train_one_hot, val_idx_tensor)  
        else:
            x_train_df = melspect_train_array[train_idx]
            x_valid_df = melspect_train_array[val_idx]
            y_train_fold = y_train_one_hot[train_idx]
            y_val_fold = y_train_one_hot[val_idx]
        j+=1
        #---------------------------------------------------------------------------
        train_generator = ImageDataGenerator().flow(x_train_df, y_train_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_generator = ImageDataGenerator().flow(x_valid_df, y_val_fold, batch_size=BATCH_SIZE, shuffle=True)
    
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
    
        start = time.time()
        history = model.fit(train_generator,
                            steps_per_epoch=len(train_generator),
                            validation_data=val_generator,
                            validation_steps=len(val_generator),
                            epochs=num_epochs,
                            verbose=1,
                            callbacks=[scheduler, checkpointer])
        end = time.time()
        print("Training time : ",(end-start))
        model_history.append(history)
    
        test_generator = ImageDataGenerator().flow(melspect_test_array, y_test_one_hot, batch_size = Batch_size_test, shuffle=False)
        predictions = model.predict(test_generator, steps=1, verbose=2)
        predicted_class_indices = np.argmax(predictions, axis=1)
        data_kfold[j] = predicted_class_indices
        gc.collect()
        
    return data_kfold, model_history
