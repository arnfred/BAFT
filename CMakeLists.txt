project(DAFT)
cmake_minimum_required(VERSION 2.4)

# ============================================================================ #
add_definitions(-DDEBUG -g -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
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
  ./src/daft.cpp
  ./src/utils.cpp
)
# ============================================================================ #
# DAFT tests
add_executable(
  test_daft
  test_daft.cpp
  ${SRCS}
)

# DAFT match tests
add_executable(
  test_daft_match
  test_daft_match.cpp
  ${SRCS}
)

target_link_libraries(
  test_daft
  ${OpenCV_LIBS}
)

target_link_libraries(
  test_daft_match
  ${OpenCV_LIBS}
)
