//! Spherical Delaunay construction through stereographic projection and Fade2D.

#include "common.hpp"

#include <Fade_2D.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

using GEOM_FADE2D::Fade_2D;
using GEOM_FADE2D::Point2;
using GEOM_FADE2D::Triangle2;

struct Vec3d {
  double x;
  double y;
  double z;
};

struct Options {
  int repeat = 1;
  int threads = 1;
  bool native_plane = false;
  bool raw_plane = false;
  bool fast = false;
  bool validate = false;
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: BENCH INPUT.f32 [--native-plane|--raw-plane] [--fast] "
        "[--validate] [--threads N] [--repeat N]");
  }
  Options options;
  for (int i = 2; i < argc;) {
    const std::string option = argv[i];
    if (option == "--native-plane") {
      options.native_plane = true;
      ++i;
      continue;
    }
    if (option == "--raw-plane") {
      options.raw_plane = true;
      ++i;
      continue;
    }
    if (option == "--fast") {
      options.fast = true;
      ++i;
      continue;
    }
    if (option == "--validate") {
      options.validate = true;
      ++i;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::runtime_error(
          "usage: BENCH INPUT.f32 [--native-plane|--raw-plane] [--fast] "
          "[--validate] [--threads N] [--repeat N]");
    }
    const int value = std::stoi(argv[i + 1]);
    if (value < 1) {
      throw std::runtime_error(option + " must be positive");
    }
    if (option == "--repeat") {
      options.repeat = value;
    } else if (option == "--threads") {
      options.threads = value;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
    i += 2;
  }
  if (options.native_plane && options.raw_plane) {
    throw std::runtime_error("--native-plane and --raw-plane are exclusive");
  }
  return options;
}

Vec3d normalize(Point3f point) {
  const Vec3d out{point.x, point.y, point.z};
  const double length =
      std::sqrt(out.x * out.x + out.y * out.y + out.z * out.z);
  if (!(length > 0.0) || !std::isfinite(length)) {
    throw std::runtime_error("input contains a non-finite or zero point");
  }
  return {out.x / length, out.y / length, out.z / length};
}

double dot(Vec3d a, Vec3d b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3d subtract(Vec3d a, Vec3d b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3d cross(Vec3d a, Vec3d b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

Vec3d scale(Vec3d value, double factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

Vec3d unit(Vec3d value) {
  const double length = std::sqrt(dot(value, value));
  if (!(length > 0.0) || !std::isfinite(length)) {
    throw std::runtime_error("degenerate spherical Delaunay triangle");
  }
  return scale(value, 1.0 / length);
}

std::array<Vec3d, 2> tangent_basis(Vec3d pole) {
  const Vec3d axis = std::abs(pole.x) < 0.8 ? Vec3d{1.0, 0.0, 0.0}
                                                 : Vec3d{0.0, 1.0, 0.0};
  const Vec3d e1 = unit(cross(pole, axis));
  return {e1, cross(pole, e1)};
}

std::array<std::uint32_t, 3> face_indices(const Triangle2* triangle) {
  std::array<std::uint32_t, 3> face{};
  for (int i = 0; i < 3; ++i) {
    const int index = triangle->getCorner(i)->getCustomIndex();
    if (index < 0) {
      throw std::runtime_error("Fade2D lost an input custom index");
    }
    face[i] = static_cast<std::uint32_t>(index);
  }
  return face;
}

Vec3d spherical_dual(const std::array<std::uint32_t, 3>& face,
                     const std::vector<Vec3d>& points) {
  const Vec3d a = points[face[0]];
  const Vec3d b = points[face[1]];
  const Vec3d c = points[face[2]];
  Vec3d normal = cross(subtract(b, a), subtract(c, a));
  if (dot(normal, a) < 0.0) {
    normal = scale(normal, -1.0);
  }
  return unit(normal);
}

void observe_face(const std::array<std::uint32_t, 3>& face,
                  const std::vector<Vec3d>& points,
                  std::vector<std::uint32_t>& degrees,
                  std::uint64_t& checksum,
                  TopologyFingerprint& topology) {
  if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2] ||
      face[0] >= points.size() || face[1] >= points.size() ||
      face[2] >= points.size()) {
    throw std::runtime_error("Fade2D produced an invalid spherical face");
  }
  for (const std::uint32_t index : face) {
    ++degrees[index];
  }
  const Vec3d dual = spherical_dual(face, points);
  checksum = hash_double(checksum, dual.x);
  checksum = hash_double(checksum, dual.y);
  checksum = hash_double(checksum, dual.z);
  topology.observe(face[0], face[1], face[2]);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto input = read_points(argv[1]);
    if (input.size() < 4) {
      throw std::runtime_error("at least four points are required");
    }
    if (input.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw std::runtime_error("point count exceeds the adapter's u32 indices");
    }

    std::vector<Vec3d> normalized;
    normalized.reserve(input.size());
    for (const Point3f point : input) {
      normalized.push_back(options.raw_plane
                               ? Vec3d{point.x, point.y, 1.0}
                               : normalize(point));
    }
    const bool planar = options.native_plane || options.raw_plane;
    Vec3d pole{};
    Vec3d e1{};
    Vec3d e2{};
    if (!planar) {
      pole = normalized.front();
      const auto basis = tangent_basis(pole);
      e1 = basis[0];
      e2 = basis[1];
    }
    const int actual_threads = GEOM_FADE2D::setGlobalNumCPU(options.threads);

    for (int iteration = 1; iteration <= options.repeat; ++iteration) {
      const auto construct_start = std::chrono::steady_clock::now();
      const auto project_start = construct_start;
      std::vector<Point2> projected;
      projected.reserve(normalized.size() - static_cast<std::size_t>(!planar));
      const std::size_t first = planar ? 0 : 1;
      for (std::size_t i = first; i < normalized.size(); ++i) {
        const Vec3d point = normalized[i];
        double u;
        double v;
        if (planar) {
          u = point.x;
          v = point.y;
        } else {
          const double denominator = 1.0 - dot(point, pole);
          if (!(denominator > 0.0) || !std::isfinite(denominator)) {
            throw std::runtime_error(
                "stereographic projection encountered the pole twice");
          }
          u = dot(point, e1) / denominator;
          v = dot(point, e2) / denominator;
        }
        if (!std::isfinite(u) || !std::isfinite(v)) {
          throw std::runtime_error(
              "projection produced a non-finite coordinate");
        }
        projected.emplace_back(u, v);
        projected.back().setCustomIndex(static_cast<int>(i));
      }
      const double project_ms = elapsed_ms(project_start);

      const auto triangulate_start = std::chrono::steady_clock::now();
      Fade_2D triangulation;
      triangulation.setFastMode(options.fast);
      triangulation.insert(projected);
      const double triangulate_ms = elapsed_ms(triangulate_start);
      const double construct_ms = elapsed_ms(construct_start);

      const auto materialize_start = std::chrono::steady_clock::now();
      std::vector<Triangle2*> planar_faces;
      triangulation.getTrianglePointers(planar_faces);
      std::vector<Point2*> hull;
      std::size_t face_count = planar_faces.size();
      if (!planar) {
        triangulation.getConvexHull(true, hull);
        if (hull.size() < 3) {
          throw std::runtime_error("Fade2D returned a degenerate convex hull");
        }

        face_count += hull.size();
        const std::size_t expected_faces = 2 * input.size() - 4;
        if (face_count != expected_faces) {
          throw std::runtime_error(
              "recovered spherical face count mismatch: expected " +
              std::to_string(expected_faces) + ", got " +
              std::to_string(face_count));
        }
      }

      std::uint64_t checksum = 0x6a09e667f3bcc909ULL;
      TopologyFingerprint topology;
      std::vector<std::uint32_t> degrees(input.size(), 0);
      for (const Triangle2* triangle : planar_faces) {
        observe_face(face_indices(triangle), normalized, degrees, checksum,
                     topology);
      }
      if (!planar) {
        for (std::size_t i = 0; i < hull.size(); ++i) {
          const int a = hull[i]->getCustomIndex();
          const int b = hull[(i + 1) % hull.size()]->getCustomIndex();
          if (a < 0 || b < 0) {
            throw std::runtime_error("Fade2D hull lost an input custom index");
          }
          observe_face(
              {0, static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)},
              normalized, degrees, checksum, topology);
        }
      }
      if (!planar) {
        for (const std::uint32_t degree : degrees) {
          if (degree < 3) {
            throw std::runtime_error(
                "recovered spherical triangulation has degree below three");
          }
          checksum = hash_mix(checksum, degree);
        }
      }
      const std::size_t incidences = 3 * face_count;
      const double materialize_ms = elapsed_ms(materialize_start);
      double validation_ms = 0.0;
      if (options.validate) {
        const auto validation_start = std::chrono::steady_clock::now();
        if (!triangulation.checkValidity(true, "benchmark validation")) {
          throw std::runtime_error("Fade2D multiprecision validation failed");
        }
        validation_ms = elapsed_ms(validation_start);
      }

      const char* backend =
          options.raw_plane    ? (options.fast ? "fade2d-raster-fast"
                                               : "fade2d-raster")
          : options.native_plane ? "fade2d-native"
          : options.fast         ? "fade2d-stereo-fast"
                                 : "fade2d-stereo";
      std::cout << "RESULT backend=" << backend << " n=" << input.size()
                << " iteration=" << iteration
                << " construct_ms=" << construct_ms
                << " project_ms=" << project_ms
                << " triangulate_ms=" << triangulate_ms
                << " materialize_ms=" << materialize_ms
                << " total_ms=" << construct_ms + materialize_ms
                << " validation_ms=" << validation_ms
                << " vertices=" << face_count << " cells=" << input.size()
                << " incidences=" << incidences << " workers=" << actual_threads
                << " fast_mode=" << options.fast
                << " validated=" << options.validate
                << " topology_sum=" << std::hex << topology.sum
                << " topology_xor=" << topology.xor_value
                << " checksum=" << checksum << std::dec << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
