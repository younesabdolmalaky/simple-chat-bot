# simple-chat-bot
# Building a Chatbot using TensorFlow
This code is a simple implementation of a Chatbot that uses TensorFlow and TensorFlow's Keras API to train the model. The code is written in Python and is built upon several libraries including Numpy, Pandas, and TensorFlow.

## Encoder-Decoder with Attention
  
Explain encoder decoder with attention mechanism and where you need a image tell me
The encoder-decoder architecture is a popular neural network design pattern used in tasks such as machine translation, text summarization, and image captioning.
The encoder part of the network takes in an input sequence, such as a sentence or an image, and processes it through a series of hidden layers to produce a fixed-length context vector that represents the input. The decoder part of the network then takes this context vector as input and generates an output sequence, such as a translated sentence or a caption for the image.
The attention mechanism is a key component of the decoder that allows the model to focus on specific parts of the input sequence when generating each output step. In practice, the attention mechanism is implemented as an additional layer that computes a weighted sum of the encoder's hidden states, where the weights represent the model's attention over the input sequence at each decoder step.
Regarding the image, the encoder-decoder architecture can be used to perform tasks such as image captioning, where the goal is to generate a natural language description of an image. The encoder part of the network can be designed to process the image data, such as the raw pixel values, and produce a context vector that represents the image. The decoder part of the network can then use the attention mechanism to focus on specific regions of the image when generating each word of the caption. This allows the model to attend to the most relevant parts of the image as it generates the description, leading to more accurate and descriptive captions.

![RNN-Encoder-Decoder-Network-with-Attention-layer-Encoder-is-a-bidirectional-LSTM-whose](https://user-images.githubusercontent.com/75095471/218437797-803f126e-1f40-4773-b567-94cb0ab49c68.png)


## code
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
