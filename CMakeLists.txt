cmake_minimum_required(VERSION 2.8.3)
project(rhoban_random)

option(RHOBAN_RANDOM_ENABLE_EXAMPLES "Building examples" OFF)
option(RHOBAN_RANDOM_ENABLE_OPENCV "Building with OpenCV" ON)

if (RHOBAN_RANDOM_ENABLE_OPENCV)
  set(OpenCV_DIR "/usr/local/include/")
  find_package(OpenCV 3.2.0 REQUIRED)
  add_definitions(-DENABLE_OPENCV)
endif ()

find_package(catkin REQUIRED COMPONENTS
  eigen rhoban_utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rhoban_random
  CATKIN_DEPENDS eigen rhoban_utils
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

set(DIRECTORIES
  src/rhoban_random
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

add_library(rhoban_random
  ${ALL_SOURCES}
)

if (RHOBAN_RANDOM_ENABLE_OPENCV)
  target_link_libraries(rhoban_random ${OpenCV_LIBS})
endif ()

set (LINKED_LIBRARIES
  ${catkin_LIBRARIES}
  rhoban_random
)

if (RHOBAN_RANDOM_ENABLE_OPENCV)
  set (LINKED_LIBRARIES
    ${LINKED_LIBRARIES}
    ${OpenCV_LIBS}
  )
endif ()

set(TESTS
  gaussian_mixture_model
  )

if (CATKIN_ENABLE_TESTING)
  # create an executable for each test
  foreach(test ${TESTS})
    string(REPLACE "/" "_" id ${test})
    catkin_add_gtest(${id} tests/${test}.cpp)
    target_link_libraries(${id} ${LINKED_LIBRARIES})
  endforeach(test)
endif()

if (RHOBAN_RANDOM_ENABLE_EXAMPLES)
  add_executable(gmm examples/gmm.cpp)
  target_link_libraries(gmm ${LINKED_LIBRARIES})
endif ()
