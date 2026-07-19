# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =========================================================
# PTODSL package wiring
# =========================================================

set(PTODSL_PACKAGE_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ptodsl")
set(PTOAS_PY_PACKAGE_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ptoas")
set(PTOAS_WHEEL_BOOTSTRAP_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/ptoas_wheel_bootstrap.py")
set(PTODSL_BUILD_ROOT "${CMAKE_BINARY_DIR}/python")
set(PTODSL_BUILD_PACKAGE_DIR
    "${PTODSL_BUILD_ROOT}/ptodsl")
set(PTOAS_PY_BUILD_PACKAGE_DIR
    "${PTODSL_BUILD_ROOT}/ptoas")

add_custom_target(PTODSLPackage ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PTODSL_BUILD_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${PTODSL_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${PTOAS_PY_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${PTODSL_PACKAGE_SRC_DIR}"
          "${PTODSL_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${PTOAS_PY_PACKAGE_SRC_DIR}"
          "${PTOAS_PY_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${PTOAS_WHEEL_BOOTSTRAP_SOURCE}"
          "${PTODSL_BUILD_ROOT}/ptoas_wheel_bootstrap.py"
  COMMENT "Staging ptodsl package into build/python"
  VERBATIM
)

install(
  DIRECTORY "${PTODSL_PACKAGE_SRC_DIR}"
  DESTINATION "."
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${PTOAS_PY_PACKAGE_SRC_DIR}"
  DESTINATION "."
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  FILES "${PTOAS_WHEEL_BOOTSTRAP_SOURCE}"
  DESTINATION "."
  COMPONENT PTOAS_Runtime
)
