FROM tensorflow/tensorflow:2.4.1-gpu 

COPY ./ /code
WORKDIR /code

RUN apt-get update \
    && pip install --upgrade pip    
# RUN apt-get install -y libsndfile-dev
# RUN pip3 install pickle5

RUN pip3 install -r requirements.txt

# ENTRYPOINT [ "python", "trainer/json_main.py" ]