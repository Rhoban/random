cmake_minimum_required(VERSION 3.16.3)
project(rhoban_random)


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

add_library(rhoban_random SHARED
  ${ALL_SOURCES}
)
target_link_libraries(rhoban_random PUBLIC Eigen3::Eigen) 
target_include_directories(rhoban_random PUBLIC 
  include
)


set(TESTS
  gaussian_mixture_model
  )

#if (CATKIN_ENABLE_TESTING)
#  # create an executable for each test
#  foreach(test ${TESTS})
#    string(REPLACE "/" "_" id ${test})
#    catkin_add_gtest(${id} tests/${test}.cpp)
#    target_link_libraries(${id} rhoban_random)
#  endforeach(test)
#endif()
