FROM tensorflow/tensorflow:2.12.0-gpu

# Install python 3.11, the base image comes with 3.8
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt upgrade -y
RUN apt-get install -y python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
RUN update-alternatives --config python3
RUN apt install -y python-is-python3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install --upgrade pip

COPY . /repo
WORKDIR /repo

RUN pip install -r requirements_docker.txt
RUN pip install tensorflow==2.12.0

EXPOSE 5001
CMD ["python", "./keras-server.py"]