
#import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os

# Print to check they all work
print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

#download dataset
cache_dir = './tmp'
dataset_file_name = 'shakespeare.txt'
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=pathlib.Path(cache_dir).absolute()
)
print("DATASET PATH IS HERE:")
print(dataset_file_path)

#reading the database file
text = open(dataset_file_path, mode = 'r').read()
print('length of text: {} characters'.format(len(text)))

#take a look at the first 250 characters in text to check it's reading the file
print(text[:350])

#take unique characters in the file
vocab = sorted(set(text))

# printing unique characters
print('{} unique characters'.format(len(vocab)))
print('vocab:', vocab)

#map characters to their indices in vocabulary
char2index = {char: index for index, char in enumerate(vocab)}

print('{')
for char, _ in zip(char2index, range(20)):
  print(' {:4s}: {:3d},'.format(repr(char), char2index[char]))
print(' ...\n}')

#map character indices to characters from vocab
index2char = np.array(vocab)
print(index2char)

#convert characters in text to indices

text_as_int = np.array([char2index[char] for char in text])

print('text_as_int length: {}'.format(len(text_as_int)))

# prints the first 15 characters of the file and then again as indices
print('{} --> {}'.format(repr(text[:15]), repr(text_as_int[:15])))





## Training Sequences

#the maximum length sentence wanted for a single input in characters
sequence_length = 100
examples_per_epoch = len(text) // (sequence_length + 1)

print ('examples_per_epoch:', examples_per_epoch)

#create training dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for char in char_dataset.take(5):
    print(index2char[char.numpy()])

#generate batched sequences out of the char_dataset
sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)

#sequence size is the same as examples_per_epoch
print('sequence count: {}'.format(len(list(sequences.as_numpy_iterator()))))
print()

#sequence examples
for item in sequences.take(5):
  print(repr(''.join(index2char[item.numpy()])))

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

#dataset size is the same as examplels_per_epoch
#but each element of a sequence now has a length of 'sequence_length'
#and not 'sequence_length + 1'
print('dataset size: {}'.format(len(list(dataset.as_numpy_iterator()))))

for input_example, target_example in dataset.take(1):
  print('Input sequence size:',repr(len(input_example.numpy())))
  print('Target sequence size:', repr(len(target_example.numpy())))
  print()
  print('Input:', repr(''.join(index2char[input_example.numpy()])))
  print('Target:', repr(''.join(index2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print('Step {:2d}'.format(i))
    print('  input: {} ({:s})'.format(input_idx, repr(index2char[input_idx])))
    print('  expected output: {} ({:s})'.format(target_idx, repr(index2char[target_idx])))






## Sequences and Batches

# Batch size.
BATCH_SIZE = 64

# Buffer size to shuffle the dataset (TF data is designed to work
# with possibly infinite sequences, so it doesn't attempt to shuffle
# the entire sequence in memory. Instead, it maintains a buffer in
# which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

print('Batched dataset size: {}'.format(len(list(dataset.as_numpy_iterator()))))

for input_text, target_text in dataset.take(1):
  print('1st batch: input_text:', input_text)
  print()
  print('1st batch: target_text:', target_text)

## Building the model
# Embedding

# Define parameters for embedding
tmp_vocab_size = 10 # number of characters in vocabluary
tmp_embeding_size = 5 # size of embedding vectors for each character
tmp_input_length = 8 # length of each input sequence
tmp_batch_size = 2 # number of sequences processed together in one batch

# Creating the model with embedding parameters
tmp_model = tf.keras.models.Sequential()
tmp_model.add(tf.keras.layers.Embedding(
  input_dim=tmp_vocab_size,
  output_dim=tmp_embeding_size,
  input_length=tmp_input_length
))

# Simulate a batch input data for testing the embedding layer
# Shape: (batch_size, input_length) -> (2, 8)
tmp_input_array = np.random.randint(
  low=0,
  high=tmp_vocab_size,
  size=(tmp_batch_size, tmp_input_length)
)
# Compile the model with a dummy loss and optimizer
tmp_model.compile('rmsprop', 'mse')
# Pass the input array through the embedding layer to get output vectors
# Output shape: (batch_size, input_length, embedding_size) -> (2, 8, 5)
tmp_output_array = tmp_model.predict(tmp_input_array)

# Print the shape and contents of the input and output arrays to check
print('tmp_input_array shape:', tmp_input_array.shape)
print('tmp_input_array:')
print(tmp_input_array)
print()
print('tmp_output_array shape:', tmp_output_array.shape)
print('tmp_output_array:')
print(tmp_output_array)

# Define parapeters for the full model
vocab_size = len(vocab) # length of the vocabulary in chars

embedding_dim = 254 # the embedding dimensions

rnn_units = 1024 # number of RNN units


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.models.Sequential()
    
        # I have added this line to fix an error
        # Explicity define the input shape for the model
        model.add(tf.keras.layers.InputLayer(batch_input_shape=[batch_size, None]))
        
        # Embedding layer: converts integer token indices to embedding vectors
        model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size, # Total number of unique tokens
            output_dim=embedding_dim, # Size of the embedding vector for each token
        ))
        
        # LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=rnn_units, # Number of LSTM units (memory size)
            return_sequences=True, # Output a sequence, not just the final state
            stateful=True, # keep state between batches
            recurrent_initializer=tf.keras.initializers.GlorotNormal() # initialise reecurrent weights
        ))
        
        # Dense layer: maps LSTM outputs to vocab-sized logits for predictions
        model.add(tf.keras.layers.Dense(vocab_size))
    
        return model

# Build model using the specific parameters
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Print a summary
model.summary()

# Plot the model structure with layer names and output shapes
tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True
)



# Trying the model
# here I am testing to make sure it all works up to here.

# Run one batch through the model to test output 
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# Show predictions logits for the first character in the first sequence
print('prediction from the 1st letter of the batch 1st sequence:')
print(example_batch_predictions[0, 0])

# Sample predicted character indices from the logits
sampled_indices = tf.random.categorical(
    logits=example_batch_predictions[0],
    num_samples=1
)

# checking shape
sampled_indices.shape

sampled_indices = tf.squeeze(
    input=sampled_indices,
    axis=-1
).numpy()

sampled_indices.shape

sampled_indices


print('Input:\n',repr(''.join(index2char[input_example_batch[0]])))
print()
print('Next character prediction:\n', repr(''.join(index2char[sampled_indices])))

for i, (input_idx, sample_idx) in enumerate(zip(input_example_batch[0], sampled_indices[:5])):
    print('Prediction {:2d}'.format(i))
    print('  input: {} ({:s})'.format(input_idx, repr(index2char[input_idx])))
    print('  next prediction: {} ({:s})'.format(target_idx, repr(index2char[sample_idx])))









## Training the model

# Now I start training the model
# Define the lose function: compares predicted logits to true labels

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
  )

  # example_batch_loss = loss(target_example_batch, example_batch_predictions)


# print('scalar_loss: ', example_batch_loss.numpy().mean())

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam_optimizer,
    loss=loss
)


#Not relevent -> attempt to make checkpoints when training module
        #Directory where the checkpoints will be saved.
        #checkpoint_dir = './tmp/checkpoints'
        #checkpoint_file = 'model_checkpoint.weights.h5'
        #checkpoint_dir = 'tmp/model_checkpoint.weights.h5'
        #checkpoint_ref = checkpoint_dir, checkpoint_file
        
        #os.makedirs(checkpoint_dir, exist_ok=True)
        #print("PATH IS HERE:")
        #print(checkpoint_dir)
        
        # Name of the checkpoint files
        #heckpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}.weights.h5')
        
        #print("PATH: ", checkpoint_prefix)
        
        #checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        #    filepath=checkpoint_prefix,
        #    save_weights_only=True
        #)



# Directory where the checkpoints will be saved.
checkpoint_dir = 'tmp/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}.weights.h5')
#THIS FILE IS FOUND HERE: C:\Users\dom10\Downloads\tmp\checkpoints

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)




# The following code was used to train the model.
# Once trained, the model worked off a saved checkpoint.
'''
#check1= 'ckpt_{epoch}.weights.h5'
#check_dir1 = os.path.dirname(check1)
#cb = tf.keras.callbacks.ModelCheckpoint(
#    check1, verbose=1, save_weights_only = True
##    )

#model.fit(x_train, y_train, epochs=2, validation_data = (x_test, y_test), callbacks = [cb])


# Execute the training
EPOCHS=40

# this code didn't end up doing what i wanted it to, replaced it with
# the code above
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#    filepath='model_checkpoint.weights.h5',
#    save_weights_only=True,
#    save_best_only=True,
#    monitor='loss',
#    verbose=1
#)


history = model.fit(
    x=dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_callback
    ]
)

# ran into some difficulty here using colab, colab took 
# 10 hours to run 30 EPOCHS before restarting when laptop
# closed
# now giving Spyder a try

# checking that with training the loss of the model
# lessened with each Epoch
def render_training_history(training_history):
    loss = training_history.history['loss']
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Training set')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.show()
    
render_training_history(history)

'''






## Generating text

#tf.compat.v1.train.latest_checkpoint(checkpoint_dir)

# 
simplified_batch_size = 1
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

#model.load_weights(tf.train.lastest_checkpoint(checkpoint_dir))

# This is the saved weights from training
model.load_weights('tmp/checkpoints/ckpt_40.weights.h5')


model.build(tf.TensorShape([simplified_batch_size, None]))

model.summary()




#running generation
def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)
    text_generated = []
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

    text_generated = []

    for char_index in range(num_generate):
        # Run the model to get predictions
        predictions = model(input_indices)
        predictions = tf.squeeze(predictions, 0)

        # Apply temperature to control randomness
        predictions = predictions / temperature

        # Sample from the distribution
        predicted_id = tf.random.categorical(
            predictions, num_samples=1
        )[-1, 0].numpy()  # corrected indexing
        # Use the predicted ID as the next input
        input_indices = tf.expand_dims([predicted_id], 0)

        # Append the predicted character
        text_generated.append(index2char[predicted_id])

# This was the origonal code from colab however I came across a lot of errors    
    #model.reset_states()
    #for char_index in range(num_generate):
    #    predictions = model(input_indices)
    #    predictions = tf.squeeze(predictions, 0)
    #    predictions = predictions / temperature
    #    predicted_id = tf.random.catagorical(
    #    predictions,
    #    num_samples=1
    #    )[-1.0].numpy()
    #    input_indices = tf.expand_dims([predicted_id],0)
    #    text_generated.append(index2char[predicted_id])
        
    return (start_string + ''.join(text_generated))

print("")
print("Shakespeare Bot")
print("")

#So the model can be run by user input I have created a loop
while True:
    print("Do you want to run the model?")
    response = input("Type 'yes' to run: ")

    if response.lower() == 'yes':
        user_prompt = input("Enter your prompt to start the model: ")
        print(generate_text(model, start_string=user_prompt))
    elif response.lower() == 'no':
        print("Goodbye!")
        break
    else:
        print("Please type 'yes' or 'no'.")
