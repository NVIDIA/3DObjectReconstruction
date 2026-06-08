FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=28
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=12
ARG OPENCV_VERSION_PATCH=0


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
    libsm6 \
    libxext6 \
    libxrender-dev \
    libblas-dev \
    libssl-dev \
    liblapack-dev \
    libflann-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libzmq3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglx-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cmake
RUN cd / &&\
    wget http://www.cmake.org/files/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
    tar xf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
    cd cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH} &&\
    ./configure &&\
    make -j$(nproc) &&\
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

# Keep pip away from touching torch
ENV PIP_NO_BINARY=torch,torchvision,torchaudio

# Install OpenCV
RUN cd / && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv_contrib && \
    mkdir -p /opencv/build && \
    cd /opencv/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
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
        -DWITH_FFMPEG=OFF \
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
        -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0 10.0 12.0" \
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
ARG PCL_VERSION=1.10.0
# Install PCL
RUN cd /tmp && \
    git clone --depth 1 --branch pcl-${PCL_VERSION} https://github.com/PointCloudLibrary/pcl && \
    mkdir -p /tmp/pcl/build && cd /tmp/pcl/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_apps=OFF \
        -DBUILD_tools=OFF \
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
        -DPCL_COMMON_WARNINGS=OFF \
        -DPCL_USE_BOOST_SHARED_PTR=ON && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/pcl

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
RUN pip3 install --no-cache-dir --ignore-installed -r /workspace/requirements.txt

# Install Python dependencies with fixed versions from uv.lock
RUN pip3 install -U pip && pip3 install --no-cache-dir \
    scikit-learn==1.6.1 scikit-image==0.25.2 --force-reinstall \
    && pip3 install --no-cache-dir --ignore-installed open3d==0.19.0 \
    && pip3 install --no-cache-dir numpy==1.26.4 joblib==1.4.2 scikit-learn==1.6.1 scikit-image==0.25.2 --force-reinstall \
    && pip3 install --no-cache-dir pyrender==0.1.45 \
    && pip3 install --no-cache-dir jupyter==1.1.1 jupyterlab notebook \
    && pip3 install --no-cache-dir pyyaml>=6.0.2
RUN pip3 install --no-cache-dir PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10
RUN pip3 install warp-lang
RUN pip3 install --no-cache-dir plyfile==1.0.3
RUN git clone https://github.com/NVIDIAGameWorks/kaolin.git /kaolin && \
    cd /kaolin && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0" python3 setup.py build && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0" python3 setup.py install && \
    cd / && rm -rf /kaolin
RUN pip3 install pytorch3d -f https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.7.0cu128-cp312-cp312-linux_x86_64.whl
RUN python3 -m pip install onnxruntime-gpu==1.23.2
#install theseus-ai without libsuitesparse-dev
RUN apt-get update && apt-get install -y libopenblas-dev
ENV BASPACHO_CUDA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBASPACHO_CUDA_ARCHS='70;75;80;86;90;100;120'"
RUN git clone https://github.com/facebookresearch/baspacho.git /baspacho
RUN cd /baspacho && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        ${BASPACHO_CUDA_ARGS} \
        -DBLA_VENDOR=OpenBLAS \
        -DBUILD_SHARED_LIBS=OFF \
        -DBASPACHO_BUILD_TESTS=OFF -DBASPACHO_BUILD_EXAMPLES=OFF \
        && cmake --build build -- -j16
RUN pip3 install cython
RUN pip3 install semantic-version==2.10.0
RUN pip3 install torchlie
RUN pip3 install torchkin
RUN pip3 install ipympl
RUN git clone https://github.com/facebookresearch/theseus.git && cd theseus &&\
BASPACHO_ROOT_DIR=/baspacho pip3 install --no-build-isolation --no-deps .


RUN pip3 install --no-build-isolation flash-attn==2.7.3

# Ensure HPC-X UCX is resolved before system UCX so libucc.so.1 can find
# ucs_config_doc_nop (HPC-X 2.22.1 built for Ubuntu 24.04 requires its own UCX)
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Pyrender / PyOpenGL: must be set before any import of OpenGL (e.g. pyrender).
ENV PYOPENGL_PLATFORM=egl
# Pyrender offscreen: use NVIDIA EGL (avoids Mesa + EGL_NOT_INITIALIZED in headless/Jupyter)
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Setup environment
ENV PATH="/bin/python3:${PATH}"
RUN alias python="/bin/python3"
RUN echo 'alias python="/bin/python3"' >> /etc/bash.bashrc && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    echo "export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:${PYTHONPATH}" >> /etc/bash.bashrc 
RUN cd / && git clone https://github.com/NVlabs/BundleSDF.git && cd BundleSDF && git checkout nvbundlesdf

RUN cp -r /BundleSDF/mycuda /customize_cuda
RUN cd /customize_cuda && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0" FORCE_CUDA=1 pip install . --no-build-isolation
    
# Setup the OpenMPI
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu
RUN ln -s /opt/hpcx/ompi /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi
RUN apt remove -y libvtk9-dev || true

# Reinstall VTK package for OpenCV compatibility
RUN apt-get update && apt-get install -y libvtk9-dev freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libzmq3-dev cppzmq-dev
# Build and install BundleTrack (copy only necessary artifacts)
RUN cp -r /BundleSDF/BundleTrack /tmp/BundleTrack
RUN cd /tmp/BundleTrack && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    # Copy only the essential built artifacts to standard locations \
    cp my_cpp*.so /usr/local/lib/python3.12/dist-packages/ && \
    cp libBundleTrack.so /usr/local/lib/ && \
    cp libMY_CUDA_LIB.so /usr/local/lib/ && \
    # Update library cache \
    ldconfig && \
    # Remove all source code and build artifacts \
    cd / && \
    rm -rf /tmp/BundleTrack && \
    rm -rf /BundleSDF/BundleTrack


# Install sam2 and roma libraries
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /sam2 && \
    cd /sam2 && \
    # Temporarily unset PIP_NO_BINARY
    pip3 install --no-build-isolation -e ".[notebooks]" && \
    cd / && rm -rf /sam2/.git

RUN cd / && git clone https://github.com/Parskatt/RoMa.git &&\
    cd RoMa && git checkout edd1b8b35a94e0d6c26c73b1bb3117db8d0b04fa && \
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
RUN pip3 install --no-deps -e src/
ENV EGL_DEVICE_ID=0
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