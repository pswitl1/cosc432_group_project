FROM python:3
ADD main.py /
ADD Classifier.py /

RUN mkdir classifier_application
COPY . /classifier_application

RUN pip install argparse nltk numpy

CMD [ "./main.py", "/classifier_application/training_data_input_file.txt" ]

