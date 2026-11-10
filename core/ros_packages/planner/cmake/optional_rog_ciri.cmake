# Optional external ROG-Map core.
#
# SUPER and ROG-Map are not vendored. CIRI for this tree lives in
# planner_core/ciri.hpp. The default build does not search for ROG-Map.
#
#   cmake -DAP_PNC_WITH_ROG_MAP=ON -DAP_PNC_ROG_MAP_PREFIX=/absolute/prefix

# CIRI itself is in-tree (planner_core/ciri.hpp). This file only links an
# external ROG-Map core. Do not copy SUPER or ROG-Map sources into the tree.
option(AP_PNC_WITH_ROG_MAP "Link an external ROG-Map core. Off by default." OFF)
set(AP_PNC_ROG_MAP_PREFIX "" CACHE PATH
  "Install prefix of an external ROG-Map core (include/ and lib/)")

set(AP_PNC_HAS_ROG_MAP FALSE)

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

function(ap_pnc_link_optional_backends target)
  if(AP_PNC_HAS_ROG_MAP)
    target_compile_definitions(${target} PUBLIC AP_PNC_HAS_ROG_MAP=1)
    target_include_directories(${target} PUBLIC
      "${AP_PNC_ROG_MAP_INCLUDE}" ${PCL_INCLUDE_DIRS})
    target_link_libraries(${target} PUBLIC
      "${AP_PNC_ROG_MAP_LIBRARY}" ${PCL_LIBRARIES})
  endif()
endfunction()
