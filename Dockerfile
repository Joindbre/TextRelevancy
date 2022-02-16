FROM python:3.8-slim
RUN mkdir /plugins
WORKDIR /plugins
ADD requirements.txt /plugins
RUN pip3 install -r requirements.txt
ADD ./classifier /plugins/classifier
ADD ./nlp /plugins/nlp
ADD app.py /plugins
ADD logger.py /plugins
CMD ["gunicorn" , "-b", "0.0.0.0:8191", "app", "--timeout", "0"]


