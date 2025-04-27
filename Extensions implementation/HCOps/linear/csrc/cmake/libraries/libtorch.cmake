include(${CMAKE_CURRENT_LIST_DIR}/../utils/run-python.cmake)

# @see "../utils/python.cmake"
run_python(
    "import torch;print(torch.utils.cmake_prefix_path)"
    PY_RESULT PY_OUTPUT PY_ERROR
)

set(PYTORCH_CMAKE_PREFIX_PATH "${PY_OUTPUT}")
message(STATUS "pyouput:   ${PY_OUTPUT}")
message(STATUS "pytorch_cmake_prefix:   ${PYTORCH_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
