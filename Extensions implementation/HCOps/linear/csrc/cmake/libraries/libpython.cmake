# @see "https://cmake.org/cmake/help/latest/module/FindPython.html"
#set(PYTHON_LIBRARY "/pacific_fs/anaconda3/envs/primv/lib/libpython3.so")
#set(PYTHON_INCLUDE_DIR "/pacific_fs/anaconda3/envs/primv/include/python3.12")
#
#message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
#message(STATUS "Python_INCLUDE_DIR: ${Python_INCLUDE_DIR}")
#
#message(STATUS "Python_LIBRARY: ${Python_LIBRARY}")

find_package(
    Python 
    REQUIRED 
    COMPONENTS 
    Interpreter 
    Interpreter Development.Module Development.Embed Development.SABIModule
    #Development
)
