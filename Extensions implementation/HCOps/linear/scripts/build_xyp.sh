# Env Variables: CC, CXX

set -e  # Exit on error

SOURCE_DIR=./csrc
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
BUILD_SHARED_LIBS=OFF
CMAKE_PREFIX_PATH=/pacific_ext/xuyunpu/app/anaconda3/envs/DiT/lib/python3.12/site-packages/torch:$CMAKE_PREFIX_PATH


while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug)
            BUILD_TYPE=$1 ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        *)
            # @todo Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cmake -S $SOURCE_DIR -B $BUILD_DIR  \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="/pacific_fs/xuyunpu/app/anaconda3/envs/DiT/lib/python3.12/site-packages/torch" \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD 
#cmake -S $SOURCE_DIR -B $BUILD_DIR  \
#    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
#    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
#    -DPython3_EXECUTABLE=$(which python3) \
#    -DPython3_INCLUDE_DIR=$(python3-config --includes | awk '{print $1}' | sed 's/-I//') \
#    -DPython3_LIBRARY=$(python3-config --ldflags | awk '{print $2}' | sed 's/-L//')  \
#    -DCMAKE_PREFIX_PATH=$(python3 -c "import sys; print(sys.prefix)")
#cmake -S $SOURCE_DIR -B $BUILD_DIR -G Ninja \
#    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
#    -DCMAKE_CXX_STANDARD=$CXX_STANDARD 

taskset -c 10-15 cmake --build $BUILD_DIR -j 4
