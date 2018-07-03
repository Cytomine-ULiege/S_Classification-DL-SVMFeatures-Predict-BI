FROM cytomineuliege/software-python3-base:latest
RUN pip install scikit-learn numpy tensorflow keras h5py hdf5 pillow
RUN git clone https://github.com/waliens/S_Classif-Deep-Features-Model-Builder.git
RUN mkdir -p /app
ADD S_Classif-Deep-Features-Model-Builder/cytomine_util.py /app/cytomine_util.py
ADD S_Classif-Deep-Features-Model-Builder/keras_util.py /app/keras_util.py
ADD run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
