cmake_minimum_required(VERSION 3.10)

project(koan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(koan koan.cpp)
add_executable(test_utils tests/test_utils.cpp)
add_executable(test_gradcheck tests/test_gradcheck.cpp)

include_directories("${PROJECT_SOURCE_DIR}/")
include_directories("${PROJECT_SOURCE_DIR}/eigen/")

target_include_directories(test_utils PUBLIC "${PROJECT_SOURCE_DIR}/extern")
target_include_directories(test_gradcheck PUBLIC "${PROJECT_SOURCE_DIR}/extern")

add_compile_options(-Wall -Wextra -Werror)

if(KOAN_ENABLE_ZIP)
  target_compile_options(koan PUBLIC -Ofast -march=native -mtune=native -DKOAN_ENABLE_ZIP)
else()
  target_compile_options(koan PUBLIC -Ofast -march=native -mtune=native)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(KOAN_ENABLE_ZIP)
  find_package(ZLIB REQUIRED)
  target_link_libraries(koan PRIVATE Threads::Threads ZLIB::ZLIB)
else()
  target_link_libraries(koan PRIVATE Threads::Threads)
endif()


install(TARGETS koan DESTINATION bin)
