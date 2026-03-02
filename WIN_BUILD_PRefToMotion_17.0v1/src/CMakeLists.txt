# Define the shared library for TransformCopy
add_library(PRefToMotion SHARED PRefToMotion.cpp)

# Platform-specific linking
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin") # macOS
    target_link_libraries(PRefToMotion PRIVATE DDImage GLEW
        "-framework QuartzCore"
        "-framework IOKit"
        "-framework CoreFoundation"
        "-framework Carbon"
        "-framework ApplicationServices"
        "-framework OpenGL"
        "-framework AGL"
    )
elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux") # Linux
    target_link_libraries(PRefToMotion PRIVATE DDImage OpenGL)
endif()

# Link to Nuke library explicitly
target_link_libraries(PRefToMotion PRIVATE "C:/Program Files/Nuke17.0v1/DDImage.lib" OpenGL32)


# Install the library to the specified location
install(TARGETS PRefToMotion DESTINATION ${CMAKE_INSTALL_PREFIX})

