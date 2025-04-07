FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    autoconf \
    automake \
    libtool \
    # readline
    libreadline6-dev \
    # zlip
    zlib1g-dev \
    # openssl
    libssl-dev \
    # opengl
    freeglut3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the repo
RUN git clone https://github.com/ritukeshbharali/jemjive-3.0.git /opt/jemjive-3.0

# Set environment variables
ENV JEMDIR=/opt/jemjive-3.0/jem-3.0
ENV JIVEDIR=/opt/jemjive-3.0/jive-3.0

# Set working directory
WORKDIR ${JEMDIR}

# Configure and build
RUN chmod +x configure && \
    ./configure && \
    make lib

# Set working directory
WORKDIR ${JIVEDIR}

# Configure and build
RUN chmod +x configure && \
    ./configure && \
    make lib
