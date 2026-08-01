cmake_minimum_required(VERSION 3.20)
project(s2_voronoi_competitors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)
find_package(Qhull REQUIRED)

add_executable(bench_cgal_sphere cgal_sphere.cpp)
target_link_libraries(bench_cgal_sphere PRIVATE CGAL::CGAL)
target_compile_options(bench_cgal_sphere PRIVATE -O3 -march=native -DNDEBUG)

add_executable(bench_qhull_sphere qhull_sphere.cpp)
target_link_libraries(bench_qhull_sphere PRIVATE Qhull::qhull_r)
target_compile_options(bench_qhull_sphere PRIVATE -O3 -march=native -DNDEBUG)
