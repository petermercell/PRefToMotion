cmake_minimum_required(VERSION 3.11)

project(PRefToMotion)

if (NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION 15.0v5)
endif()

# Update the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_SHARED_LIBRARY_PREFIX "")

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
endif()

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(NDKDIR /Applications/Nuke${NUKE_VERSION}/Nuke${NUKE_VERSION}.app/Contents/MacOS)
endif ()

link_directories(${NDKDIR} ${NDKDIR}/../Frameworks)
include_directories(${NDKDIR}/include headers)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(gizmos)

# Enable code coverage flags if COVERAGE is specified
option(COVERAGE "Enable coverage reporting" ON)
if(COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Building with code coverage support")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
endif()

# Add a custom target to generate coverage report
if(COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --capture --directory . --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${LCOV} --list coverage.info
            COMMAND ${GENHTML} coverage.info --output-directory out
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
            VERBATIM
        )
    else()
        message(WARNING "lcov/genhtml not found. Cannot generate coverage report.")
    endif()
endif()
