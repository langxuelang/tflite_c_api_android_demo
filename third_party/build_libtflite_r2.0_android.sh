#!/bin/sh
set -e
#set -x

export TENSORFLOW_VER=r2.0
export TENSORFLOW_DIR=`pwd`/tensorflow

export ANDROID_NDK_HOME=${HOME}/Android/android-ndk-r20b
export ANDROID_NDK_API_LEVEL="27"
export ANDROID_BUILD_TOOLS_VERSION="29.0.3"
export ANDROID_SDK_API_LEVEL="27"
export ANDROID_SDK_HOME=${HOME}/Android/Sdk
export ANDROID_API_LEVEL="27"


git clone https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DIR}

cd ${TENSORFLOW_DIR}
git checkout ${TENSORFLOW_VER}


# install Bazel 0.26.1
#wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
#chmod 755 bazel-0.26.1-installer-linux-x86_64.sh
#sudo ./bazel-0.26.1-installer-linux-x86_64.sh

# clean up bazel cache, just in case.
bazel clean

echo "----------------------------------------------------"
echo " (configure) need to configure Android NDK/SDK.     "
echo "----------------------------------------------------"
echo "  configure ./WORKSPACE for Android builds? : y"
echo "  Android NDK to use          : /home/username/Android/android-ndk-r20b"
echo "  Android NDK API level to use: 27"
echo "  Android SDK to use          : /home/username/Android/Sdk"
echo "  Android SDK API level to use: 27"
echo "  Android build tools version : 29.0.3"
echo "----------------------------------------------------"
./configure


bazel build -s -c opt --cxxopt='--std=c++11' --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
bazel build -s -c opt --cxxopt='--std=c++11' --config android_arm64 //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so

ls -l bazel-genfiles/tensorflow/lite/
ls -l bazel-genfiles/tensorflow/lite/delegates/gpu

