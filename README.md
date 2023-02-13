# simple-chat-bot
# Building a Chatbot using TensorFlow
This code is a simple implementation of a Chatbot that uses TensorFlow and TensorFlow's Keras API to train the model. The code is written in Python and is built upon several libraries including Numpy, Pandas, and TensorFlow.

The code first imports all the necessary libraries and defines several functions to preprocess the data and load it into the model. The preprocessing functions include:

### unicode_to_ascii:
 This function takes in a string and returns an ASCII representation of the string by removing any diacritical marks.
preprocess_sentence: This function takes in a sentence and preprocesses it by converting it to lowercase, removing extra spaces, and adding the start and end tokens.
### max_length: This function returns the maximum length of a tensor.
### tokenize:
This function tokenizes the text and returns the tensor and a tokenizer for the text.
### load_dataset:
This function reads in the data from a file and preprocesses it using the above functions.
The code then loads the data and splits it into training and testing sets using train_test_split from the sklearn.model_selection module.

The data is then fed into a tf.data.Dataset object and batched for training.

The main model is an Encoder-Decoder architecture where an Encoder network is used to encode the input data into a hidden state and a Decoder network is used to decode the hidden state into the target data. The Encoder network is defined as a custom Keras model subclass called Encoder and it consists of an embedding layer and a GRU layer. The Encoder class takes in the vocabulary size, embedding dimension, and number of units for the GRU layer as parameters and initializes the network.

The model is then trained on the training data using fit method from Keras. After training, the model is evaluated on the test data to measure its performance.
