import os
import sys
import json
import nltk
import time
import datetime
import numpy as np
from nltk.stem.lancaster import LancasterStemmer


class Classifier:
    """
    class Classifier
    takes training data input and allows user to call classify on new sentences
    """

    # static variables
    ERROR_THRESHOLD = 0.5
    stemmer = LancasterStemmer()

    def __init__(self, input_file, synapse_file, hidden_neurons=10, alpha=1, epochs=50000, dropout=False,
                 dropout_percent=0.5):
        """
        Using input file, determine training data and train neural net
        :param input_file: file path to input file
        """

        # define class variables
        self.words = []
        self.classes = []
        self.documents = []
        self.training_data = []
        self.training = []
        self.output = []
        self.synapse_file = synapse_file

        # verify input file is valid
        if os.path.exists(input_file):
            if os.path.isdir(input_file):
                sys.exit('input file: %s is a directory, must be a file' % input_file)
            else:
                print('Classifier using %s as training data input file' % input_file)
        else:
            print('input file: %s does not exist' % input_file)

        # parse input
        self.parse_input_file(input_file)
        print('%s sentences in training data' % len(self.training_data))

        # set words classes and documents
        self.set_words_classes_documents()
        print('%i documents' % len(self.documents))
        print('%i classes: %s' % (len(self.classes), self.classes))
        print('%i unique stemmed words: %s' % (len(self.words), self.words))

        # set training and output
        self.create_training_data()
        print('first documents words:')
        print([Classifier.stemmer.stem(word.lower()) for word in self.documents[0][0]])
        print('first training data: %s' % self.training[0])
        print('first training output: %s' % self.output[0])

        # train and time training
        start_time = time.time()
        self.train(hidden_neurons=hidden_neurons,
                   alpha=alpha,
                   epochs=epochs,
                   dropout=dropout,
                   dropout_percent=dropout_percent)
        elapsed_time = time.time() - start_time
        print("processing time: %s seconds" % elapsed_time)

        # load our calculated synapse values
        with open(self.synapse_file) as data_file:
            synapse = json.load(data_file)
            self.synapse_0 = np.asarray(synapse['synapse0'])
            self.synapse_1 = np.asarray(synapse['synapse1'])

    def parse_input_file(self, input_file):
        """
        parse the training data input file to set training data
        :param input_file: training data input file
        """
        with open(input_file, 'r') as fin:
            for line in fin.readlines():
                line = line[line.index('{'):line.rindex('}')+1]
                self.training_data.append(json.loads(line))

    def set_words_classes_documents(self):
        """
        from training data, create words, classes, and documents
        """
        ignore_words = ['?']
        # loop through each sentence in our training data
        for pattern in self.training_data:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['sentence'])
            # add to our words list
            self.words.extend(w)
            # add to documents in our corpus
            self.documents.append((w, pattern['class']))
            # add to our classes list
            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        # stem and lower each word and remove duplicates
        self.words = [Classifier.stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
        self.words = list(set(self.words))

        # remove duplicates
        self.classes = list(set(self.classes))

    def create_training_data(self):
        """
        use classes, words, and documents to create training and training output
        """
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [Classifier.stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            self.training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)

    def train(self, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
        """
        train the neural net based on training data and save to self.synapse_file
        :param hidden_neurons: number of hidden neurons
        :param alpha: alpha
        :param epochs: number of epochs
        :param dropout: whether to dropout or not
        :param dropout_percent: percentage to dropout
        """
        x = np.array(self.training)
        y = np.array(self.output)

        print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
            hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(x), len(x[0]), 1, len(self.classes)))
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2 * np.random.random((len(x[0]), hidden_neurons)) - 1
        synapse_1 = 2 * np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(epochs + 1)):
            print('epoch: %i' % j)
            # Feed forward through layers 0, 1, and 2
            layer_0 = x
            layer_1 = Classifier.sigmoid(np.dot(layer_0, synapse_0))

            if dropout:
                layer_1 *= np.random.binomial([np.ones((len(x), hidden_neurons))], 1 - dropout_percent)[0] * (
                        1.0 / (1 - dropout_percent))

            layer_2 = Classifier.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                    break

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * Classifier.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * Classifier.sigmoid_output_to_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if j > 0:
                synapse_0_direction_count += np.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'words': self.words,
                   'classes': self.classes
                   }

        with open(self.synapse_file, 'w') as fout:
            json.dump(synapse, fout, indent=4, sort_keys=True)
        print("saved synapses to:", self.synapse_file)

    def classify(self, sentence, show_details=False):

        results = self.think(sentence, show_details)

        results = [[i, r] for i, r in enumerate(results) if r > Classifier.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

    def think(self, sentence, show_details=False):
        x = self.bow(sentence.lower(), show_details)
        if show_details:
            print("sentence:", sentence, "\n bow:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = Classifier.sigmoid(np.dot(l0, self.synapse_0))
        # output layer
        l2 = Classifier.sigmoid(np.dot(l1, self.synapse_1))
        return l2

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    @staticmethod
    def clean_up_sentence(sentence):
        """
        clean up input sentence
        :param sentence: input sentence
        :return: cleaned up sentence
        """
        Classifier.stemmer = LancasterStemmer()
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [Classifier.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    @staticmethod
    def sigmoid(x):
        """
        # compute sigmoid nonlinearity
        :param x: item to calculate on
        :return: sigmoid of x
        """
        output = 1 / (1 + np.exp(-x))
        return output

    @staticmethod
    def sigmoid_output_to_derivative(output):
        """
        convert output of sigmoid function to its derivative
        :param output: output of sigmoid function
        :return: sigmoid function of outputs derivative
        """
        return output * (1 - output)
