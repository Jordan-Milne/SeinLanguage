from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time

import pandas as pd

################Turning the lines into vectors #########################
df = pd.read_csv('data/scripts.csv')
# Making a list of lists of every line jerry says
jer = df[df['Character'] == 'JERRY']['Dialogue'].to_list()
# Flattening every Jerry line into 1 large list
jerry = ' '.join(jer)

# Selecting the unique characters from every lines Jerry says
characters_ = sorted(set(jerry))

char_vec = {u:i for i, u in enumerate(characters_)}
vec_char = np.array(characters_)
text_id = np.array([char_vec[c] for c in jerry])

################Setting Parameters for the model#########################

seq_length = 100
examples_per_epoch = len(jerry)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_id)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1024
vocab_size = len(characters_)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



########## Building the model#################

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='orthogonal'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(characters_),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# setting the loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)


####### making training checkpoints ########
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

## Training the model
# history = model.fit(dataset, epochs=30, callbacks=[checkpoint_callback])




checkpoint_dir = './training_checkpoints'
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_string):


  # Lenth of Jerry's generated line
  num_generate = 150

  # Vectorizing starting string
  input_eval = [char_vec[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []

  # Predictability
  temperature = 0.1

  model.reset_states()

  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(vec_char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model,"When will you"))



 # "I can't beleive when I was a kid couldn't have been more than that clown. Why don't you get him some tickets or something? Where are they already? Can I go? Really? You don't seem that desperate? Let me put you in a hotel. You'll be fine. Where are you going? There's more food down the hall. Oh yeah. Yeah, I know. "
