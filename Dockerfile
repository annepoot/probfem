# Stage 1: base part
# Only this part should get pushed to docker hub
FROM continuumio/miniconda3:latest AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # basics
    git \
    build-essential \
    autoconf \
    automake \
    libtool \
    # readline
    libreadline6-dev \
    # zlib
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

# Set working directory
WORKDIR /workspace

# Copy environment file
COPY ENVIRONMENT.yml .

# Create environment
RUN conda env create -f ENVIRONMENT.yml && conda clean -afy

# Stage 2a: jupyter part
# This part is compiled separately, because it depends on the state of the repo
FROM base AS jupyter

# latex dependencies for matplotlib
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    dvipng \
    ghostscript

# Set working directory
WORKDIR /workspace

# Ensure the env is always activated
SHELL ["conda", "run", "-n", "probfem", "/bin/bash", "-c"]

# Copy the full project
COPY . .

# Compile C++ backend inside the conda environment
RUN cd fem/jive/src && make && cd ../../..

# Ensure the conda environment is active at runtime
RUN echo "conda activate probfem" >> ~/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace"' >> ~/.bashrc
ENV PATH="/opt/conda/envs/probfem/bin:${PATH}"
ENV PYTHONPATH=/workspace

# Expose Jupyter's default port
EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "probfem", "jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
