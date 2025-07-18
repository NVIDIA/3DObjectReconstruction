FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=25
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=11
ARG OPENCV_VERSION_PATCH=0

ARG PCL_VERSION_MAJOR=1
ARG PCL_VERSION_MINOR=10
ARG PCL_VERSION_PATCH=0

ARG PYBIND11_VERSION_MAJOR=2
ARG PYBIND11_VERSION_MINOR=13
ARG PYBIND11_VERSION_PATCH=0

ARG YAML_CPP_VERSION_MAJOR=0
ARG YAML_CPP_VERSION_MINOR=8
ARG YAML_CPP_VERSION_PATCH=0

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libblas-dev \
    libssl-dev \
    liblapack-dev \
    gfortran \
    gnupg \
    software-properties-common \
    libflann-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libzmq3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Insttall cmake
RUN cd / &&\
    wget http://www.cmake.org/files/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
    tar xf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
    cd cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH} &&\
    ./configure &&\
    make &&\
    make install

# Install Eigen
RUN cd / && \
    wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}/eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    tar xf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    cd eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd / && \
    rm -rf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}

RUN pip3 install -U torch==2.6.0 torchvision==0.21.0

# Install OpenCV
RUN cd / && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv_contrib && \
    mkdir -p /opencv/build && \
    cd /opencv/build && \
    cmake ..  -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA_STUBS=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_MATLAB=OFF \
        -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF \
        -DCUDA_FAST_MATH=ON \
        -DMKL_WITH_OPENMP=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_OPENMP=ON \
        -DWITH_QT=ON \
        -DWITH_OPENEXR=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_FFMPEG=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DENABLE_CXX11=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DOPENCV_DNN_OPENCL=OFF \
        -DWITH_CUDA=ON \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
        -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0" \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DINSTALL_PKGCONFIG=ON \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DPKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /opencv /opencv_contrib

# Install PCL
RUN cd / && \
    git clone --depth 1 --branch pcl-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}.${PCL_VERSION_PATCH} https://github.com/PointCloudLibrary/pcl && \
    mkdir -p /pcl/build && \
    cd /pcl/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_apps=OFF \
        -DBUILD_GPU=OFF \
        -DBUILD_CUDA=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_global_tests=OFF \
        -DBUILD_simulation=OFF \
        -DCUDA_BUILD_EMULATION=OFF \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DPCL_ENABLE_SSE=ON \
        -DPCL_SHARED_LIBS=ON \
        -DWITH_VTK=OFF \
        -DPCL_ONLY_CORE_POINT_TYPES=ON \
        -DPCL_COMMON_WARNINGS=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pcl

# Install Pybind11
RUN cd / && \
    git clone --depth 1 --branch v${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH} https://github.com/pybind/pybind11 && \
    mkdir -p /pybind11/build && \
    cd /pybind11/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pybind11

# Install YAML-CPP
RUN cd / && \
    git clone --depth 1 --branch ${YAML_CPP_VERSION_MAJOR}.${YAML_CPP_VERSION_MINOR}.${YAML_CPP_VERSION_PATCH} https://github.com/jbeder/yaml-cpp && \
    mkdir -p /yaml-cpp/build && \
    cd /yaml-cpp/build && \
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DINSTALL_GTEST=OFF \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_BUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /yaml-cpp

# Create workspace directory
WORKDIR /workspace

COPY src/requirements.txt /workspace/
# Install Python dependencies
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Install Python dependencies
RUN pip3 install -U pip && pip3 install --no-cache-dir \
    scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir --ignore-installed open3d \
    && pip3 install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html \
    && pip3 install --no-cache-dir --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html \
    && pip3 install --no-cache-dir numpy==1.26.4 scipy joblib scikit-learn scikit-image --force-reinstall \
    && pip3 install --no-cache-dir pyrender \
    && pip3 install --no-cache-dir jupyter jupyterlab notebook

# Setup environment
ENV PATH="/bin/python3:${PATH}"
RUN alias python="/bin/python3"
RUN echo 'alias python="/bin/python3"' >> /etc/bash.bashrc && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    echo "export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:${PYTHONPATH}" >> /etc/bash.bashrc

RUN cd / && git clone https://github.com/NVLabs/BundleSDF.git

RUN cp -r /BundleSDF/mycuda /customize_cuda
RUN cd /customize_cuda && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" FORCE_CUDA=1 pip install . --no-build-isolation
    
# Setup the OpenMPI
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu
RUN ln -s /opt/hpcx/ompi /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi
RUN apt remove -y libvtk9-dev || true

# Reinstall VTK package for OpenCV compatibility
RUN apt-get update && apt-get install -y libvtk9-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install BundleTrack (copy only necessary artifacts)
RUN cp -r /BundleSDF/BundleTrack /tmp/BundleTrack
RUN cd /tmp/BundleTrack && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    # Copy only the essential built artifacts to standard locations \
    cp my_cpp*.so /usr/local/lib/python3.10/dist-packages/ && \
    cp libBundleTrack.so /usr/local/lib/ && \
    cp libMY_CUDA_LIB.so /usr/local/lib/ && \
    # Update library cache \
    ldconfig && \
    # Remove all source code and build artifacts \
    cd / && \
    rm -rf /tmp/BundleTrack

RUN cd / && \
    git clone https://github.com/NVlabs/FoundationStereo.git 

ENV PYTHONPATH=/FoundationStereo/core:$PYTHONPATH

# Install sam2 and roma libraries
RUN cd / && git clone https://github.com/facebookresearch/sam2.git &&\
    cd sam2 &&\
    SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]" && \ 
    python3 setup.py build_ext --inplace

RUN cd / && git clone https://github.com/Parskatt/RoMa.git &&\
    cd RoMa &&\
    pip3 install . &&\
    cd / && rm -rf /RoMa

# Final cleanup and ldconfig
RUN ldconfig && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    # Remove any remaining temporary files \
    rm -rf /tmp/*

WORKDIR /workspace

# Copy the entire package structure for proper installation
COPY src /workspace/3d-object-reconstruction/src
COPY README.md /workspace/3d-object-reconstruction/
COPY notebooks /workspace/3d-object-reconstruction/notebooks
COPY data /workspace/3d-object-reconstruction/data

# Install the package in editable mode (package files are now under src/)
WORKDIR /workspace/3d-object-reconstruction
RUN pip3 install -e src/

# Create Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Set the default command to start Jupyter notebook
CMD ["jupyter", "notebook", "--notebook-dir=/workspace/3d-object-reconstruction", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

