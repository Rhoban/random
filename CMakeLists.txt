cmake_minimum_required(VERSION 2.8.3)
project(rhoban_random)

# OpenCV required for Expectation Maximization
find_package(OpenCV 3.2.0 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  eigen
  rhoban_utils
  tclap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


catkin_package(
  INCLUDE_DIRS include ${OpenCV_INCLUDE_DIRS}
  LIBRARIES rhoban_random ${OpenCV_LIBS}
  CATKIN_DEPENDS eigen rhoban_utils tclap
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
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
target_link_libraries(rhoban_random ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(expectation_maximization examples/expectation_maximization.cpp)
target_link_libraries(expectation_maximization rhoban_random ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(generate_gmm_samples examples/generate_gmm_samples.cpp)
target_link_libraries(generate_gmm_samples rhoban_random ${catkin_LIBRARIES} ${OpenCV_LIBS})

set(TESTS
  gaussian_mixture_model
  )

if (CATKIN_ENABLE_TESTING)
  # create an executable for each test
  foreach(test ${TESTS})
    string(REPLACE "/" "_" id ${test})
    catkin_add_gtest(${id} tests/${test}.cpp)
    target_link_libraries(${id} ${catkin_LIBRARIES} rhoban_random ${OpenCV_LIBS})
  endforeach(test)
endif()
