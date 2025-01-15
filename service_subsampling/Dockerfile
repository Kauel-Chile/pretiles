FROM python:bookworm

WORKDIR /home/opengeo

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    python3-software-properties \
    git \
    build-essential \
    cmake \
    g++ \
    libboost-all-dev \
    cmake-curses-gui \
    gcc \
    libeigen3-dev \
    wget \
    libcurl4-openssl-dev \
    libproj-dev \
    libsqlite3-dev \
    libgeos-dev \
    libtbb-dev \
    libtiff-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    liblzma-dev \
    libxml2-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clona e instala LAStools
RUN git clone https://github.com/m-schuetz/LAStools.git \
    && cd LAStools/LASzip \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make

# Clona e instala PotreeConverter
RUN git clone https://github.com/potree/PotreeConverter.git \
    && cd PotreeConverter \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DLASZIP_INCLUDE_DIRS=/home/opengeo/LAStools/LASzip/dll -DLASZIP_LIBRARY=/home/opengeo/LAStools/LASzip/build/src/liblaszip.so .. \
    && make 

# Crea un enlace simbólico a la librería LASzip para asegurarse de que esté accesible
RUN ln -s /home/opengeo/LAStools/LASzip/build/src/liblaszip.so /usr/lib

COPY . /home/opengeo/service_subsampling

WORKDIR /home/opengeo/service_subsampling

RUN pip install -r requirement.txt

EXPOSE 8000

CMD ["python", "main.py"]