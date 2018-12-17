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
    parser.add_argument('--hidden-neurons', help='number of hidden neurons in training', type=int, default=6)
    parser.add_argument('--alpha', help='alpha in training', type=float, default=.03)
    parser.add_argument('--epoch', help='number of epochs in training', type=int, default=200)
    parser.add_argument('--dropout', help='dropout boolean for training', action='store_true')
    parser.add_argument('--dropout-percent', help='dropout boolean for training', type=float, default=0.0)
    parser.add_argument('--use-stopwords', help='use stopwords', action='store_true')
    parser.add_argument('--cp2', help='run classifier for cp2, which classifies based on all classes', action='store_true')
    parser.add_argument('--use-db-class', help='include database sentences', action='store_true')

    args = parser.parse_args()

    # set up classifier
    classifier = Classifier(input_file=args.input,
                            synapse_file=args.synapse_file,
                            hidden_neurons=args.hidden_neurons,
                            alpha=args.alpha,
                            epochs=args.epoch,
                            dropout=args.dropout,
                            dropout_percent=args.dropout_percent,
                            use_stopwords=args.use_stopwords,
                            cp2=args.cp2,
                            use_db_class=args.use_db_class)

    percent_correct = classifier.classify_test_data()

    return percent_correct

if __name__ == '__main__':
    main()
