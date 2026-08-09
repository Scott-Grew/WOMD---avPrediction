FROM --platform=linux/amd64 python:3.10-slim

RUN pip install --no-cache-dir waymo-open-dataset-tf-2-12-0==1.6.7

WORKDIR /work

CMD ["python3", "-c", "import waymo_open_dataset; import tensorflow; print('waymo-open-dataset ok · tensorflow', tensorflow.__version__)"]
