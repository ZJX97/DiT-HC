set(LIB_NAME "kpprim")
set(KBLAS_LIB_INFO "-L/pacific_fs/HPCKit/25.3.30/kml/bisheng-0303/lib/sme/kblas/multi -lkblas")
file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

Python_add_library(${LIB_NAME} SHARED ${CXX_SOURCES})
add_library(${PROJECT_NAMESPACE}${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
    ${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    ${LIB_NAME} 
    PUBLIC 
    ${TORCH_LIBRARIES}
    ${KBLAS_LIB_INFO} 
)

install_libraries(
    ${LIB_NAME}
    ${PROJECT_NAMESPACE}
    ${LIB_NAME}
)
