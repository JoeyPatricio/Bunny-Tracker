cmake_minimum_required(VERSION 3.25)
project(bunnycam_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# libonnxruntime.so.1 is copied next to the executables (below).
set(CMAKE_BUILD_RPATH "$ORIGIN")

set(ONNXRUNTIME_ROOT "$ENV{HOME}/.cache/bunnycam/onnxruntime-1.29.0" CACHE PATH
    "ONNX Runtime 1.29.0 from scripts/fetch-onnxruntime.sh")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. Run scripts/fetch-onnxruntime.sh")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
  IMPORTED_SONAME "libonnxruntime.so.1"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
configure_file("${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so" "${CMAKE_BINARY_DIR}/libonnxruntime.so.1" COPYONLY)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
# cpp-httplib is header-only, and not every distro ships its CMake config.
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)
find_package(Catch2 3 REQUIRED)

add_library(agent_core STATIC
  src/motion.cpp
  src/decision.cpp
  src/inference.cpp
  src/ffmpeg.cpp
  src/server.cpp)
target_include_directories(agent_core PUBLIC src ${HTTPLIB_INCLUDE_DIR})
target_compile_options(agent_core PUBLIC -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(agent_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(agent_core PUBLIC
  onnxruntime nlohmann_json::nlohmann_json OpenSSL::SSL OpenSSL::Crypto Threads::Threads)

add_executable(bunnycam-agent src/main.cpp)
target_link_libraries(bunnycam-agent PRIVATE agent_core)

enable_testing()
include(Catch)
foreach(name motion decision inference ffmpeg server labels)
  add_executable(test_${name} tests/test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE agent_core Catch2::Catch2WithMain)
  target_compile_definitions(test_${name} PRIVATE BUNNYCAM_REPO_DIR="${CMAKE_SOURCE_DIR}/..")
  catch_discover_tests(test_${name})
endforeach()
