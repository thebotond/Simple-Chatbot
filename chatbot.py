#If you wish to retrain the model delete all files except "chatbot.py" and "topics.json"
#nltk has a dependency on contrib from tensorflow which has been removed.
#For this reason the program will run using python 3.6 and tensorflow 1.14 or 1.15

import nltk, tflearn, random, json, pickle, os
import numpy as np
import tensorflow as tf

from os import path
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer

engStem = EnglishStemmer()

#open the responses file
with open("topics.json") as response_file:
    data = json.load(response_file)

#look for training data, set up nn for training
try:
    with open("trainingdata.pickle", "rb") as training_file:
        words, labels, training, output = pickle.load(training_file)
      
except:
    words = []
    labels = []
    docs_words = []
    docs_tags = []

    for topic in data["topics"]:
        for pattern in topic["patterns"]:
            tempWords = nltk.word_tokenize(pattern)
            words.extend(tempWords)
            docs_words.append(tempWords)
            docs_tags.append(topic["tag"])

        if topic["tag"] not in labels:
            labels.append(topic["tag"])

    words = [engStem.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_words):
        pool = []

        tempWord = [engStem.stem(w.lower()) for w in doc]

        for w in words:
            if w in tempWord:
                pool.append(1)
            else:
                pool.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_tags[x])] = 1

        training.append(pool)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)

    with open("trainingdata.pickle", "wb") as training_file:
        pickle.dump((words, labels, training, output), training_file)
        

tf.reset_default_graph()

neuronet = tflearn.input_data(shape=[None, len(training[0])]) #input layer
neuronet = tflearn.fully_connected(neuronet, 8) #hidden layer with 8 neurons
neuronet = tflearn.fully_connected(neuronet, 8) #hidden layer with 8 neurons
neuronet = tflearn.fully_connected(neuronet, len(output[0]), activation='Softmax') #output layer with softmax activation function
neuronet = tflearn.regression(neuronet)

model = tflearn.DNN(neuronet)

#train the model if no trained weights and biases found - epoch value was decided after trial and error
if path.exists('checkpoint'):
    model.load('model.tflearn')
else:
    model.fit(training, output, n_epoch=750, batch_size=8, show_metric=True)
    model.save('model.tflearn')

#generate a numpy array for the user input
def word_pool(userInput, words):
    pool = [0 for _ in range(len(words))]

    user_words = nltk.word_tokenize(userInput)
    user_words = [engStem.stem(word.lower()) for word in user_words]

    for se in user_words:
        for indx, ws in enumerate(words):
            if ws == se:
                pool[indx] = 1
            
    return np.array(pool)


def chat():
    print("Type 'quit' or 'exit' to exit the program.")
    while True:
        userInput = input()
        if userInput.lower() == "quit" or userInput.lower() == "exit":
            break

        predictions = model.predict([word_pool(userInput, words)]) #predict using the user input and words parameters
        predictions_index = np.argmax(predictions) #retrieve the position of the highest likelyhood in the array
        tag = labels[predictions_index] #select a tag at the position, response is chosen randomly
        
        if np.amax(predictions) > 0.8: #if the input makes "a lot of" sense, respond with a random  response from the chosen tag (topic)
            for tagData in data["topics"]:
                if tagData['tag'] == tag:
                    responses = tagData['responses']
            print(random.choice(responses))

        else: #if the machine can't make sense of what the user said, advise
            print("I'm sorry, I did not understand that.")
chat()
