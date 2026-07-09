cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
  else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
  endif()
endif()

if(NOT DEFINED ASCEND_DRIVER_PATH)
  if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
  else()
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
  endif()
endif()

find_program(BISHENG_EXECUTABLE bisheng HINTS "${ASCEND_HOME_PATH}/bin")
if(NOT BISHENG_EXECUTABLE)
  message(FATAL_ERROR "bisheng compiler not found")
endif()

set(CMAKE_CXX_COMPILER "${BISHENG_EXECUTABLE}" CACHE FILEPATH "" FORCE)

project(rope_cce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(rope_cce SHARED rope_cce_kernel.cpp)

target_include_directories(
  rope_cce
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/cce"
    "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/pkg_inc/profiling"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
    "${ASCEND_DRIVER_PATH}/kernel_inc"
)

target_compile_options(
  rope_cce
  PRIVATE
    -O3
    -std=gnu++17
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -Wno-unknown-attributes
    -fPIC
    -xcce
    -Xhost-start
    -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    --cce-aicore-arch=dav-c310-vec
    --cce-simd-vf-fusion=false
    -DREGISTER_BASE
)

target_link_options(
  rope_cce
  PRIVATE
    --cce-fatobj-link
    -Wl,-soname,librope_cce.so
)
