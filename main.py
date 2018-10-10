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
    parser.add_argument('--synapse-file', '-s', help='input file containing training data', required=False, type=str,\
                        default='synapses.json')
    parser.add_argument('--quiet', '-q', help='surpress logging', action='store_true')
    args = parser.parse_args()

    # set up classifier
    classifier = Classifier(args.input, args.synapse_file)

    # try to classify some test sentences
    classifier.classify("Users information must be kept private")
    classifier.classify("The Application should have a modern look")
    classifier.classify("The Application should have circular buttons")
    classifier.classify("The application must run fast and smoothly")
    classifier.classify("The users information must be linked to an emergency contact")
    classifier.classify("The application should be designed with code readability in mind")

if __name__ == '__main__':
    main()
