#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kpops::kpprim" for configuration "Release"
set_property(TARGET kpops::kpprim APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kpops::kpprim PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "Python::Python"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkpprim.so"
  IMPORTED_SONAME_RELEASE "libkpprim.so"
  )

list(APPEND _cmake_import_check_targets kpops::kpprim )
list(APPEND _cmake_import_check_files_for_kpops::kpprim "${_IMPORT_PREFIX}/lib/libkpprim.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
