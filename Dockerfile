FROM python:3.9-slim

LABEL maintainer="academic-implementation"
LABEL description="Scheme4: Digital Twin Network Security Framework"

# Preserve proxy environment variables (passed from external)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG ALL_PROXY
ARG all_proxy

# Set proxy environment variables
ENV HTTP_PROXY=${HTTP_PROXY:-""}
ENV HTTPS_PROXY=${HTTPS_PROXY:-""}
ENV http_proxy=${http_proxy:-""}
ENV https_proxy=${https_proxy:-""}
ENV ALL_PROXY=${ALL_PROXY:-""}
ENV all_proxy=${all_proxy:-""}
ENV NO_PROXY=localhost,127.0.0.1,host.docker.internal,hubproxy.docker.internal

# Install system dependencies (required for PBC library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    libssl-dev \
    git \
    automake \
    autoconf \
    libtool \
    flex \
    bison \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Compile and install PBC library from source
RUN cd /tmp && \
    wget https://crypto.stanford.edu/pbc/files/pbc-0.5.14.tar.gz && \
    tar -xzf pbc-0.5.14.tar.gz && \
    cd pbc-0.5.14 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/pbc-*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install charm-crypto (from GitHub)
RUN git clone --depth=1 https://github.com/JHUISI/charm.git /tmp/charm && \
    cd /tmp/charm && \
    git submodule init && \
    git submodule update && \
    ./configure.sh && \
    pip install . && \
    cd / && rm -rf /tmp/charm

# Install other Python dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create persistence directory
RUN mkdir -p /app/dt_persistence

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]