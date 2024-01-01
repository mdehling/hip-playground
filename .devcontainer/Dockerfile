ARG VERSION="2023.11.02"
FROM docker.io/mdehling/ubuntu-cuda-dev:${VERSION}

ARG kr_url="https://repo.radeon.com/rocm/rocm.gpg.key"
ARG kr_dir="/etc/apt/keyrings"
ARG kr_path="${kr_dir}/rocm.gpg"
RUN mkdir --parents --mode=0755 ${kr_dir} && \
    wget ${kr_url} -O - | gpg --dearmor > ${kr_path}

ARG rocm_repo="https://repo.radeon.com/rocm/apt/6.0"
RUN echo "deb [arch=amd64 signed-by=${kr_path}] ${rocm_repo} jammy main" \
    | tee -a /etc/apt/sources.list.d/rocm.list && \
    echo "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" \
    | tee /etc/apt/preferences.d/rocm-pin-600
RUN apt-get update

RUN apt-get -y install rocm-hip-sdk rocblas

RUN echo "/opt/rocm/lib\n/opt/rocm/lib64" | tee -a /etc/ld.so.conf.d/rocm.conf
RUN ldconfig

ENV PATH=$PATH:/opt/rocm/bin:/usr/local/cuda/bin
