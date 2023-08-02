# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop using the LEAF frontend."""

import os
from typing import Optional

import gin
from leaf_audio import models
#from example import data
import tensorflow as tf
#import tensorflow_datasets as tfds

#----------------------------------- added ------------------------------------
import sys
sys.path.append('C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/git_leaf/leaf-audio/leaf_audio')
from OnlineDataToDF import DatatoDF
from CustomVoiceLeaf import CustomVoiceLeaf

labels = 'Mass Other'.split()
dir_train_oversampled = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/70_20_10/oversampled/train'
dir_val_oversampled = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/70_20_10/oversampled/Val'
dir_test_processed ='C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/OutputOnline/1.e/70_20_10/padded/NewTest'

#dataset_train_df = oversampled_train_df,
#dataset_val_df = oversampled_val_df,


@gin.configurable
def train(workdir: str = 'C:/Users/zahra/Desktop/FarzanehFiles/Codes/2_Code_TT_AugPercentAllTrain_TV/OnlineAugmentation/git_leaf/SavedModels',
          train_dir = dir_train_oversampled,
          val_dir = dir_val_oversampled,
          num_epochs: int = 10,
          steps_per_epoch: Optional[int] = None,
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          num_classes: int = num_classes_tr, 
          **kwargs):
  """Trains a model on a dataset.

  Args:
    workdir: where to store the checkpoints and metrics.
    dataset: name of a tensorflow_datasets audio datasset.
    num_epochs: number of epochs to training the model for.
    steps_per_epoch: number of steps that define an epoch. If None, an epoch is
      a pass over the entire training set.
    learning_rate: Adam's learning rate.
    batch_size: size of the mini-batches.
    **kwargs: arguments to the models.AudioClassifier class, namely the encoder
      and the frontend models (tf.keras.Model).
  """
 

  oversampled_train_df, num_classes_tr = DatatoDF(dir_train_oversampled, labels)
  oversampled_val_df, num_classes_val = DatatoDF(dir_val_oversampled, labels) 
  new_test_df, num_classes_test = DatatoDF(dir_test_processed, labels) 

  dataset_train_df = oversampled_train_df,
  dataset_val_df = oversampled_val_df,

  train_aug_gen = CustomVoiceLeaf(train_dir , data_df = dataset_train_df, num_classes= num_classes_tr, batch_size = batch_size, shuffle = True, augment = False)
  val_aug_gen = CustomVoiceLeaf(val_dir, data_df = dataset_val_df, num_classes= num_classes_val, batch_size = batch_size, shuffle = True, augment = False)
  
  
  model = models.AudioClassifier(num_outputs=num_classes, **kwargs)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = 'sparse_categorical_accuracy'
  model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[metric])

  ckpt_path = os.path.join(workdir, 'checkpoint')
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=ckpt_path,
      save_weights_only=True,
      monitor=f'val_{metric}',
      mode='max',
      save_best_only=True)

  model.fit(x=train_aug_gen[0][0], y=train_aug_gen[0][1],
            validation_data=(val_aug_gen[0][0],val_aug_gen[0][1]),
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[model_checkpoint_callback])
