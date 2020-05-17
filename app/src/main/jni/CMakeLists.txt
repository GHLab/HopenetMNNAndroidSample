# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(pathPROJECT ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
message("CMAKE_CURRENT_SOURCE_DIR:${CMAKE_CURRENT_SOURCE_DIR}")
message("pathPROJECT:${pathPROJECT}")

set(pathJNILIBS ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

set(pathLIBMNN ${pathJNILIBS}/libMNN.so)
message("pathLIBMNN: ${pathLIBMNN}")

set(pathLIBMNN_EXPRESS ${pathJNILIBS}/libMNN_Express.so)
message("pathLIBMNN_EXPRESS: ${pathLIBMNN_EXPRESS}")

set(pathLIBMNN_CL ${pathJNILIBS}/libMNN_CL.so)
message("pathLIBMNN_CL: ${pathLIBMNN_CL}")

set(pathLIBMNN_VULKAN ${pathJNILIBS}/libMNN_Vulkan.so)
message("pathLIBMNN_VULKAN: ${pathLIBMNN_VULKAN}")

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

include_directories(${pathPROJECT}/src/main/jni/include)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

SET(SOURCES
        ${pathPROJECT}/src/main/jni/HeadPoseDetectorJNI.cpp
        ${pathPROJECT}/src/main/jni/HeadPoseDetector.cpp)

add_library(MNNSampleLib SHARED ${SOURCES})
set_target_properties(MNNSampleLib PROPERTIES LINKER_LANGUAGE CXX)

add_library(lib_mnn SHARED IMPORTED)
set_target_properties(lib_mnn PROPERTIES IMPORTED_LOCATION ${pathLIBMNN})

add_library(lib_mnn_exp SHARED IMPORTED)
set_target_properties(lib_mnn_exp PROPERTIES IMPORTED_LOCATION ${pathLIBMNN_EXPRESS})

add_library(lib_mnn_cl SHARED IMPORTED)
set_target_properties(lib_mnn_cl PROPERTIES IMPORTED_LOCATION ${pathLIBMNN_CL})

add_library(lib_mnn_vulkan SHARED IMPORTED)
set_target_properties(lib_mnn_vulkan PROPERTIES IMPORTED_LOCATION ${pathLIBMNN_VULKAN})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        MNNSampleLib
        lib_mnn
        lib_mnn_exp
        lib_mnn_cl
        lib_mnn_vulkan
        jnigraphics

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib} )