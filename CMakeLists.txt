project(Bary)
cmake_minimum_required( VERSION 2.6 )

#
# Open Computer Vision required!
#

find_package(OpenCV)
if (NOT OpenCV_FOUND )
	message(FATAL_ERROR "Package OpenCV not found.")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )
set( LIBS ${OpenCV_LIBRARIES} )

set(SRCS
    tcv.h
    tcv_simd.h
    tcv_barycentric.h
    tcv_barycentric.cpp
    imagewarp.cpp
    colorfill.cpp
    main.cpp
)

add_executable(Bary ${SRCS} )	
target_link_libraries(Bary ${LIBS} )
