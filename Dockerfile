FROM neubiaswg5/ml-keras-base

RUN pip install pillow

# Add utilities files
RUN git clone https://github.com/waliens/S_Classif-Deep-Features-Model-Builder.git
RUN mkdir /app
RUN cp ./S_Classif-Deep-Features-Model-Builder/cytomine_util.py /app/cytomine_util.py
RUN cp ./S_Classif-Deep-Features-Model-Builder/keras_util.py /app/keras_util.py

ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
