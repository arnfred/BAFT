project(BAFT)
cmake_minimum_required(VERSION 2.4)

# ============================================================================ #
add_definitions(-DDEBUG -g -O3 -Wall -std=c++11 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
#add_definitions(-DDEBUG -g -Wall -std=c++11 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# ============================================================================ #
# OpenCV
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})

include_directories(
   ${OpenCV_INCLUDE_DIRS}
)

link_directories(
   ${OpenCV_LIBRARY_DIRS}
)

# ============================================================================ #
set(SRCS
  ./src/baft.cpp
  ./src/utils.cpp
)
# ============================================================================ #
# BAFT tests
add_executable(
  test_baft
  test_baft.cpp
  ${SRCS}
)

# BAFT match tests
add_executable(
  test_baft_match
  test_baft_match.cpp
  ${SRCS}
)

# Write out descriptors
add_executable(
  write_descriptors
  write_descriptors.cpp
  ${SRCS}
)

target_link_libraries(
  test_baft
  ${OpenCV_LIBS}
)

target_link_libraries(
  test_baft_match
  ${OpenCV_LIBS}
)

target_link_libraries(
  write_descriptors
  ${OpenCV_LIBS}
)
