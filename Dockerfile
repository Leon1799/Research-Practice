# Use an official PyTorch runtime as a parent image. 
FROM tensorflow/tensorflow:2.21.0-gpu

# Set the working directory in the container; use your account name to replace 'your_account' below 
WORKDIR /home/leon

# Install cuDNN 9 required by TensorFlow (tensorflow docker image contained cuDNN 8 instead)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*


# install requirements
COPY requirements_311.txt .

RUN pip install --no-cache-dir -r requirements_311.txt

# Copy the current directory contents into the container; use your account name to replace 'your_account' below
COPY . /home/leon

# Make port 8888 available to the world outside this container, in case you need to run jupyter notebook
EXPOSE 8888



