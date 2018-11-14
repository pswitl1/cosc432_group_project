#!/usr/bin/env python

from Classifier import Classifier
from argparse import ArgumentParser


def main():
    """
    test driver for Classifier class
    """

    # parse arguments
    parser = ArgumentParser('driver for the Classifier class')
    parser.add_argument('input', help='input file containing training data', type=str)
    parser.add_argument('--synapse-file', '-s', help='input file containing training data (this will skip training)', required=False, type=str,\
                        default='')
    parser.add_argument('--quiet', '-q', help='surpress logging', action='store_true')
    parser.add_argument('--hidden-neurons', help='number of hidden neurons in training', type=int, default=7)
    parser.add_argument('--alpha', help='alpha in training', type=float, default=.07)
    parser.add_argument('--epoch', help='number of epochs in training', type=int, default=1000)
    parser.add_argument('--dropout', help='dropout boolean for training', action='store_true')
    parser.add_argument('--dropout-percent', help='dropout boolean for training', type=float, default=0.1)
    parser.add_argument('--disable-stopwords', help='dont use stopwords', action='store_true')


    args = parser.parse_args()
#train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

    # set up classifier
    classifier = Classifier(input_file=args.input,
                            synapse_file=args.synapse_file,
                            hidden_neurons=args.hidden_neurons,
                            alpha=args.alpha,
                            epochs=args.epoch,
                            dropout=args.dropout,
                            dropout_percent=args.dropout_percent,
                            disable_stopwords=args.disable_stopwords)

    # try to classify some test sentences
    classifier.classify("Users information must be kept private.")
    classifier.classify("The Application should have a modern look.")
    classifier.classify("The Application should have circular buttons.")
    classifier.classify("The application must run fast and smoothly.")
    classifier.classify("The users information must be linked to an emergency contact.")
    classifier.classify("This will present you with a list of all of the contacts currently contained in your Address Book.")


if __name__ == '__main__':
    main()
