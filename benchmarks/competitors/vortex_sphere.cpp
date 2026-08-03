// Common-input adapter for philipclaude/vortex's spherical Voronoi backend.

#include <voronoi.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint64_t mix(uint64_t hash, uint64_t value) {
  return hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));
}

std::vector<double> read_points(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("could not open input");
  const auto size = input.tellg();
  if (size <= 0 || size % 12 != 0)
    throw std::runtime_error("input must contain packed f32 xyz triples");
  input.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  input.read(bytes.data(), size);
  if (!input) throw std::runtime_error("failed to read all input points");

  const size_t n = bytes.size() / 12;
  std::vector<double> sites(3 * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t d = 0; d < 3; ++d) {
      float value;
      std::memcpy(&value, bytes.data() + 12 * i + 4 * d, sizeof(value));
      sites[3 * i + d] = value;
    }
  }
  return sites;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::cerr << "usage: bench_vortex_sphere INPUT [--threads 1|16] [--full] "
                 "[--repeat N]\n";
    return 2;
  }
  std::string input_path = argv[1];
  size_t threads = 1;
  size_t repeat = 1;
  bool full = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--full") {
      full = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      threads = std::stoul(argv[++i]);
    } else if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::stoul(argv[++i]);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if ((threads != 1 && threads != 16) || repeat == 0)
    throw std::runtime_error("threads must be 1 or 16 and repeat must be positive");

  auto sites = read_points(input_path);
  const size_t n = sites.size() / 3;
  vortex::SphereDomain domain;

  for (size_t iteration = 1; iteration <= repeat; ++iteration) {
    vortex::VoronoiDiagramOptions options;
    options.verbose = false;
    options.parallel = threads != 1;
    options.store_mesh = full;
    options.store_facet_data = full;
    options.store_delaunay_triangles = false;
    options.neighbor_algorithm = vortex::NearestNeighborAlgorithm::kSphereQuadtree;

    const auto begin = Clock::now();
    std::vector<vortex::index_t> order(n);
    vortex::sort_points_on_zcurve(sites.data(), n, 3, order);
    std::vector<double> reordered(3 * n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t d = 0; d < 3; ++d)
        reordered[3 * i + d] = sites[3 * order[i] + d];
    }
    vortex::VoronoiDiagram diagram(3, reordered.data(), n);
    diagram.create_sqtree(-1);
    diagram.compute(domain, options);
    const double construct_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();

    const auto materialize_begin = Clock::now();
    uint64_t checksum = 0x082efa98ec4e6c89ULL;
    size_t incidences = 0;
    size_t failures = 0;
    for (auto status : diagram.status()) {
      checksum = mix(checksum, static_cast<uint8_t>(status));
      failures += status != vortex::VoronoiStatusCode::kSuccess;
    }
    if (full) {
      for (size_t i = 0; i < diagram.vertices().n(); ++i) {
        const auto* vertex = diagram.vertices()[i];
        for (int d = 0; d < diagram.vertices().dim(); ++d) {
          uint64_t bits;
          std::memcpy(&bits, vertex + d, sizeof(bits));
          checksum = mix(checksum, bits);
        }
      }
      for (size_t i = 0; i < diagram.polygons().n(); ++i) {
        const auto* polygon = diagram.polygons()[i];
        const size_t count = diagram.polygons().length(i);
        incidences += count;
        checksum = mix(checksum, count);
        for (size_t j = 0; j < count; ++j) checksum = mix(checksum, polygon[j]);
      }
    }
    const double materialize_ms = std::chrono::duration<double, std::milli>(
                                      Clock::now() - materialize_begin)
                                      .count();
    std::cout << "RESULT backend=" << (full ? "vortex" : "vortex-construct")
              << " n=" << n << " iteration=" << iteration
              << " construct_ms=" << construct_ms
              << " materialize_ms=" << materialize_ms
              << " total_ms=" << construct_ms + materialize_ms
              << " vertices=" << diagram.vertices().n()
              << " cells=" << diagram.polygons().n()
              << " incidences=" << incidences << " failures=" << failures
              << " checksum=" << std::hex << checksum << std::dec << '\n';
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "bench_vortex_sphere: " << error.what() << '\n';
  return 1;
}
