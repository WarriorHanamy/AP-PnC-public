# Optional external ROG-Map core and ROS-free CIRI.
#
# SUPER, ROG-Map, and CIRI are not vendored. Their file headers are commonly
# LGPL-3.0, the standalone ROG-Map LICENSE is GPL-3.0, and this repository is
# BSD-3-Clause. Until that combination is confirmed in writing, the default
# build does not search for, download, or compile those sources.
#
#   cmake -DAP_PNC_WITH_ROG_MAP=ON -DAP_PNC_ROG_MAP_PREFIX=/absolute/prefix
#   cmake -DAP_PNC_WITH_CIRI=ON -DAP_PNC_CIRI_PREFIX=/absolute/prefix
#
# The CIRI prefix must provide a ROS-free header `ciri/corridor.hpp` declaring
# `ciri::decompose_segment(...)` as documented in infra/SUPER_SCENE.md.
# SUPER's own `super_core/ciri.h` includes a ROS interface and is not accepted.

option(AP_PNC_WITH_ROG_MAP "Link an external ROG-Map core. Off by default." OFF)
option(AP_PNC_WITH_CIRI "Link an external ROS-free CIRI. Off by default." OFF)
set(AP_PNC_ROG_MAP_PREFIX "" CACHE PATH
  "Install prefix of an external ROG-Map core (include/ and lib/)")
set(AP_PNC_CIRI_PREFIX "" CACHE PATH
  "Install prefix of an external ROS-free CIRI (include/ciri and lib/)")

set(AP_PNC_HAS_ROG_MAP FALSE)
set(AP_PNC_HAS_CIRI FALSE)

if(AP_PNC_WITH_ROG_MAP)
  if(AP_PNC_ROG_MAP_PREFIX STREQUAL "" AND DEFINED ENV{AP_PNC_ROG_MAP_PREFIX})
    set(AP_PNC_ROG_MAP_PREFIX "$ENV{AP_PNC_ROG_MAP_PREFIX}")
  endif()
  find_path(AP_PNC_ROG_MAP_INCLUDE rog_map/rog_map.h
    PATHS "${AP_PNC_ROG_MAP_PREFIX}/include" NO_DEFAULT_PATH)
  find_library(AP_PNC_ROG_MAP_LIBRARY NAMES rog_map
    PATHS "${AP_PNC_ROG_MAP_PREFIX}/lib" "${AP_PNC_ROG_MAP_PREFIX}/lib64"
    NO_DEFAULT_PATH)
  find_package(PCL REQUIRED COMPONENTS common)
  if(NOT AP_PNC_ROG_MAP_INCLUDE OR NOT AP_PNC_ROG_MAP_LIBRARY)
    message(FATAL_ERROR
      "AP_PNC_WITH_ROG_MAP=ON but rog_map/rog_map.h or librog_map was not found "
      "under AP_PNC_ROG_MAP_PREFIX (${AP_PNC_ROG_MAP_PREFIX}). Do not copy "
      "SUPER or ROG-Map sources into this repository. See infra/SUPER_SCENE.md.")
  endif()
  set(AP_PNC_HAS_ROG_MAP TRUE)
endif()

if(AP_PNC_WITH_CIRI)
  if(AP_PNC_CIRI_PREFIX STREQUAL "" AND DEFINED ENV{AP_PNC_CIRI_PREFIX})
    set(AP_PNC_CIRI_PREFIX "$ENV{AP_PNC_CIRI_PREFIX}")
  endif()
  find_path(AP_PNC_CIRI_INCLUDE ciri/corridor.hpp
    PATHS "${AP_PNC_CIRI_PREFIX}/include" NO_DEFAULT_PATH)
  find_library(AP_PNC_CIRI_LIBRARY NAMES ciri_core ciri
    PATHS "${AP_PNC_CIRI_PREFIX}/lib" "${AP_PNC_CIRI_PREFIX}/lib64"
    NO_DEFAULT_PATH)
  if(NOT AP_PNC_CIRI_INCLUDE OR NOT AP_PNC_CIRI_LIBRARY)
    message(FATAL_ERROR
      "AP_PNC_WITH_CIRI=ON but ciri/corridor.hpp or libciri_core was not found "
      "under AP_PNC_CIRI_PREFIX (${AP_PNC_CIRI_PREFIX}). SUPER's ROS CIRI "
      "header is not accepted. See infra/SUPER_SCENE.md.")
  endif()
  set(AP_PNC_HAS_CIRI TRUE)
endif()

function(ap_pnc_link_optional_backends target)
  if(AP_PNC_HAS_ROG_MAP)
    target_compile_definitions(${target} PUBLIC AP_PNC_HAS_ROG_MAP=1)
    target_include_directories(${target} PUBLIC
      "${AP_PNC_ROG_MAP_INCLUDE}" ${PCL_INCLUDE_DIRS})
    target_link_libraries(${target} PUBLIC
      "${AP_PNC_ROG_MAP_LIBRARY}" ${PCL_LIBRARIES})
  endif()
  if(AP_PNC_HAS_CIRI)
    target_compile_definitions(${target} PUBLIC AP_PNC_HAS_CIRI=1)
    target_include_directories(${target} PUBLIC "${AP_PNC_CIRI_INCLUDE}")
    target_link_libraries(${target} PUBLIC "${AP_PNC_CIRI_LIBRARY}")
  endif()
endfunction()
