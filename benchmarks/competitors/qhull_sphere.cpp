#include "common.hpp"

extern "C" {
#include <libqhull_r/libqhull_r.h>
}

#include <cstdio>
#include <limits>

int main(int argc, char** argv) {
  try {
    const int repeat = parse_repeat(argc, argv);
    const auto input = read_points(argv[1]);
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Qhull point count exceeds its int API");
    }

    std::vector<coordT> coordinates;
    coordinates.reserve(input.size() * 3);
    for (const auto& point : input) {
      coordinates.push_back(point.x);
      coordinates.push_back(point.y);
      coordinates.push_back(point.z);
    }

    for (int iteration = 1; iteration <= repeat; ++iteration) {
      qhT state;
      qhT* qh = &state;
      qh_zero(qh, stderr);
      char flags[] = "qhull Qt";

      const auto construct_start = std::chrono::steady_clock::now();
      const int exit_code = qh_new_qhull(
          qh, 3, static_cast<int>(input.size()), coordinates.data(), false,
          flags, nullptr, stderr);
      const double construct_ms = elapsed_ms(construct_start);
      if (exit_code != qh_ERRnone) {
        qh_freeqhull(qh, !qh_ALL);
        throw std::runtime_error("Qhull construction failed with code " +
                                 std::to_string(exit_code));
      }

      const auto materialize_start = std::chrono::steady_clock::now();
      std::uint64_t checksum = 0x13198a2e03707344ULL;
      std::size_t facets = 0;
      std::size_t incidences = 0;
      facetT* facet;
      vertexT* vertex;
      vertexT** vertexp;
      FORALLfacets {
        if (!facet->simplicial) {
          continue;
        }
        ++facets;
        checksum = hash_double(checksum, facet->normal[0]);
        checksum = hash_double(checksum, facet->normal[1]);
        checksum = hash_double(checksum, facet->normal[2]);
        FOREACHvertex_(facet->vertices) {
          checksum = hash_mix(checksum,
                              static_cast<std::uint64_t>(qh_pointid(qh, vertex->point)));
          ++incidences;
        }
      }
      const double materialize_ms = elapsed_ms(materialize_start);

      print_result("qhull", input.size(), iteration, construct_ms, materialize_ms,
                   facets, input.size(), incidences, checksum);
      qh_freeqhull(qh, !qh_ALL);
      int remaining_long = 0;
      int remaining_short = 0;
      qh_memfreeshort(qh, &remaining_long, &remaining_short);
      if (remaining_long != 0 || remaining_short != 0) {
        throw std::runtime_error("Qhull reported unfreed allocations");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
