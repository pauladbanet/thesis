FROM tensorflow/tensorflow:2.4.1-gpu 

COPY ./ /code
WORKDIR /code

RUN apt-get update
RUN apt-get install -y libsndfile-dev
RUN pip3 install pickle5

RUN pip3 install -r requirements.txt

