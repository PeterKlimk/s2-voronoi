#include "common.hpp"

#include <CGAL/Delaunay_triangulation_on_sphere_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Projection_on_sphere_traits_3.h>

#include <cmath>

int main(int argc, char** argv) {
  try {
    const int repeat = parse_repeat(argc, argv);
    const auto input = read_points(argv[1]);

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Traits = CGAL::Projection_on_sphere_traits_3<Kernel>;
    using Triangulation = CGAL::Delaunay_triangulation_on_sphere_2<Traits>;
    using Point = Kernel::Point_3;

    std::vector<Point> points;
    points.reserve(input.size());
    for (const auto& point : input) {
      points.emplace_back(point.x, point.y, point.z);
    }

    for (int iteration = 1; iteration <= repeat; ++iteration) {
      const auto construct_start = std::chrono::steady_clock::now();
      Triangulation triangulation(points.begin(), points.end(), Point(0, 0, 0), 1.0);
      const double construct_ms = elapsed_ms(construct_start);

      const auto materialize_start = std::chrono::steady_clock::now();
      std::uint64_t checksum = 0x243f6a8885a308d3ULL;
      std::size_t incidences = 0;
      for (auto face = triangulation.solid_faces_begin();
           face != triangulation.solid_faces_end(); ++face) {
        const auto dual = triangulation.dual(face);
        checksum = hash_double(checksum, CGAL::to_double(dual.x()));
        checksum = hash_double(checksum, CGAL::to_double(dual.y()));
        checksum = hash_double(checksum, CGAL::to_double(dual.z()));
        incidences += 3;
      }
      for (auto vertex = triangulation.vertices_begin();
           vertex != triangulation.vertices_end(); ++vertex) {
        auto face = triangulation.incident_faces(vertex);
        const auto first = face;
        std::size_t degree = 0;
        if (face != nullptr) {
          do {
            if (!triangulation.is_ghost(face)) {
              ++degree;
            }
          } while (++face != first);
        }
        checksum = hash_mix(checksum, degree);
      }
      const double materialize_ms = elapsed_ms(materialize_start);

      print_result("cgal", input.size(), iteration, construct_ms, materialize_ms,
                   triangulation.number_of_solid_faces(),
                   triangulation.number_of_vertices(), incidences, checksum);
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
