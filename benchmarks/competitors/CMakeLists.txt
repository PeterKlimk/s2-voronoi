cmake_minimum_required(VERSION 3.20)
project(s2_voronoi_competitors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL QUIET)
find_package(Qhull QUIET)

set(FADE2D_ROOT "" CACHE PATH "Path to an extracted Fade2D release")
set(FADE2D_LIBRARY "" CACHE FILEPATH "Path to libfade2d")

if(CGAL_FOUND)
  add_executable(bench_cgal_sphere cgal_sphere.cpp)
  target_link_libraries(bench_cgal_sphere PRIVATE CGAL::CGAL)
  target_compile_options(bench_cgal_sphere PRIVATE -O3 -march=native -DNDEBUG)
endif()

if(Qhull_FOUND)
  add_executable(bench_qhull_sphere qhull_sphere.cpp)
  target_link_libraries(bench_qhull_sphere PRIVATE Qhull::qhull_r)
  target_compile_options(bench_qhull_sphere PRIVATE -O3 -march=native -DNDEBUG)
endif()

if(FADE2D_ROOT)
  find_path(
    FADE2D_INCLUDE_DIR
    Fade_2D.h
    PATHS "${FADE2D_ROOT}/include_fade2d"
    NO_DEFAULT_PATH
    REQUIRED
  )
  if(NOT FADE2D_LIBRARY)
    find_library(
      FADE2D_DISCOVERED_LIBRARY
      fade2d
      PATHS "${FADE2D_ROOT}"
      PATH_SUFFIXES
        lib_ubuntu24.04_x86_64
        lib_ubuntu22.04_x86_64
        lib_ubuntu20.04_x86_64
        lib_rhel8.9_x86_64
        lib_rhel8.8_x86_64
        lib_opensuse_leap_15.6_x86_64
        lib_opensuse_leap_15.5_x86_64
      NO_DEFAULT_PATH
      REQUIRED
    )
    set(FADE2D_LIBRARY "${FADE2D_DISCOVERED_LIBRARY}")
  endif()

  add_executable(bench_fade2d_sphere fade2d_sphere.cpp)
  target_include_directories(bench_fade2d_sphere PRIVATE "${FADE2D_INCLUDE_DIR}")
  target_link_libraries(bench_fade2d_sphere PRIVATE "${FADE2D_LIBRARY}")
  target_compile_options(bench_fade2d_sphere PRIVATE -O3 -march=native -DNDEBUG)
endif()
