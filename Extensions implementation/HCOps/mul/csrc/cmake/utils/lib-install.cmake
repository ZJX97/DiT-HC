include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# @breif Install target libraries
function(install_libraries LIB_COMPONENT LIB_NAMESPACE LIB_LIST)
    message(STATUS "Libraries to install:")
    message(STATUS "|- Component: ${LIB_COMPONENT}")
    message(STATUS "|- Namespace: ${LIB_NAMESPACE}")
    message(STATUS "|- Library Targets: [ ${LIB_LIST} ]")

    # Install libraries
    install(
        TARGETS ${LIB_LIST}
        COMPONENT ${LIB_COMPONENT}
        EXPORT ${LIB_COMPONENT}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION Lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    # Install headers
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        COMPONENT ${LIB_COMPONENT}
        DESTINATION include
    )

    # Install cmake config files
    install(
        EXPORT ${LIB_COMPONENT}Targets
        COMPONENT ${LIB_COMPONENT}
        FILE ${LIB_COMPONENT}Targets.cmake
        NAMESPACE ${LIB_NAMESPACE}
        DESTINATION lib/cmake/${LIB_COMPONENT}
    )

    # Generate and install version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Generate and install config file
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/config.cmake.in/${LIB_COMPONENT}-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}-config.cmake
        INSTALL_DESTINATION lib/cmake/${LIB_COMPONENT}
    )

    # Install config files
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}-config-version.cmake
        COMPONENT ${LIB_COMPONENT}
        DESTINATION 
            lib/cmake/${LIB_COMPONENT}
    )
endfunction()
