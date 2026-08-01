#pragma once

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Point3f {
  float x;
  float y;
  float z;
};

inline std::vector<Point3f> read_points(const std::string& path) {
  const auto bytes = std::filesystem::file_size(path);
  if (bytes == 0 || bytes % sizeof(Point3f) != 0) {
    throw std::runtime_error("input must contain packed little-endian f32 xyz triples");
  }
  std::vector<Point3f> points(bytes / sizeof(Point3f));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("failed to read all input points");
  }
  return points;
}

inline double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

inline std::uint64_t hash_mix(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

inline std::uint64_t hash_double(std::uint64_t hash, double value) {
  return hash_mix(hash, std::bit_cast<std::uint64_t>(value));
}

inline int parse_repeat(int argc, char** argv) {
  if (argc == 2) {
    return 1;
  }
  if (argc == 4 && std::string(argv[2]) == "--repeat") {
    const int repeat = std::stoi(argv[3]);
    if (repeat > 0) {
      return repeat;
    }
  }
  throw std::runtime_error("usage: BENCH INPUT.f32 [--repeat N]");
}

inline void print_result(const char* backend, std::size_t n, int iteration,
                         double construct_ms, double materialize_ms,
                         std::size_t vertices, std::size_t cells,
                         std::size_t incidences, std::uint64_t checksum) {
  std::cout << "RESULT"
            << " backend=" << backend << " n=" << n
            << " iteration=" << iteration
            << " construct_ms=" << construct_ms
            << " materialize_ms=" << materialize_ms
            << " total_ms=" << construct_ms + materialize_ms
            << " vertices=" << vertices << " cells=" << cells
            << " incidences=" << incidences << " checksum=" << std::hex
            << checksum << std::dec << '\n';
}
