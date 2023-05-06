# ShieldTalk

```
# ShieldTalk

# Introduction:
The issue of online harassment, cyberbullying, and toxicity has become increasingly prevalent in recent years. As more and more people rely on online platforms for communication and social interaction, it is important to create a safe and welcoming environment for all users.

# Objective
Our objective in this hackathon is to:
- The objective of making a toxic comment classification system is to automatically identify and flag inappropriate, offensive, or harmful comments in online platforms such as social media, discussion forums, and blogs

# Dataset
We used the **[Jigsaw Database](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** from kaggle
This dataset contains about 160,000 data entries in which there's column, id, sentence, and 6 labels i.e.(toxicity, severe_toxicity, obscene, threat, insult, identity attack)

# Models
You can find the trained model in the link [here](https://drive.google.com/drive/folders/1Nd2zWO9KqaTG5cOWdh7i04O6Jva1P3lI).
# Model Description
We have created a Deep Learning Model consisting of bidirectional lstm instead of normal lstm because in bidirectional, our input flows in two directions, making a bi-lstm different from the regular LSTM.

We used Bidirectional LSTMs for toxicity classification because they capture the context and dependencies in text data, which can be difficult to model with other models.

![image](https://user-images.githubusercontent.com/54539708/236645488-86196eeb-f8ee-43be-bfd5-c439787fd1bd.png)

# Model Architechture
1.  The first layer is a Bidirectional Long Short-Term Memory (LSTM) layer with 128 units and return_sequences=True. This layer is used to capture the context and long-term dependencies in the input data, which is important for text classification tasks. The Bidirectional wrapper allows the layer to process the input in both forward and backward directions, which improves the model's ability to capture the relationships between words.
2.  The second layer is a GlobalMaxPooling1D layer, which extracts the maximum value from each feature map of the previous layer. This layer reduces the dimensionality of the output and provides a global view of the input sequence, which can help the model to identify the most important features.
3.  The third layer is a Dropout layer with a rate of 0.1, which randomly drops out 10% of the input units during training. This layer is used to prevent overfitting, which can occur when the model learns to memorize the training data instead of generalizing to new data.
4.  The fourth layer is a Dense layer with 256 units and a Rectified Linear Unit (ReLU) activation function. This layer is used to map the extracted features to a higher-level representation that can be used to classify the input text. The ReLU activation function helps to introduce non-linearity into the model, which can improve its ability to learn complex patterns.
5.  The fifth layer is another Dropout layer with a rate of 0.1, which is used to further prevent overfitting.
6.  The sixth and final layer is a Dense layer with 6 units and a Sigmoid activation function. This layer is used to output the probability of each class label, where the Sigmoid activation function ensures that the output probabilities are between 0 and 1. The binary_crossentropy loss function is used because this is a multi-label classification problem.

![image](https://user-images.githubusercontent.com/54539708/236645497-45489922-3596-4e9a-94a4-d216667a895f.png)

# Methodology
1. Datai.e. Video and Text is uploaded to the website
-  If an audio is passed
i)  There's a function to get audio from the video
ii) The audio is then passed to a function that transcribes the audio and converts into text with the help of Whisper model
iii) The text is then preprocessed and fed to the model, then the prediction is taken and plotted in form of bargraph in the website
- If an Text is passed
i) The function takes the text preprocesses it, and is fed to the model, then the prediction is taken and plotted in the form of bar graph in the website

```
