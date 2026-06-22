// Exact-predicate 3D convex hull probe for S2 Delaunay reference checks.
//
// Input:  whitespace-separated rows on stdin:
//   <u32-id> <x> <y> <z>
//
// Output: one sorted hull triangle per line:
//   <u32-id-a> <u32-id-b> <u32-id-c>
//
// Build:
//   g++ -O3 -std=c++17 scripts/cgal_hull3.cpp -lgmp -lmpfr -o /tmp/cgal_hull3

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/convex_hull_3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;

struct Key {
    std::uint64_t x;
    std::uint64_t y;
    std::uint64_t z;

    bool operator==(const Key &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct KeyHash {
    std::size_t operator()(const Key &key) const {
        std::uint64_t h = key.x;
        h ^= key.y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= key.z + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

static std::uint64_t bits(double v) {
    if (v == 0.0) {
        v = 0.0;
    }
    std::uint64_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

static Key key_for(double x, double y, double z) {
    return Key{bits(x), bits(y), bits(z)};
}

static Key key_for(const Point &p) {
    return key_for(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Point> points;
    std::unordered_map<Key, std::uint32_t, KeyHash> ids;

    std::uint32_t id;
    double x;
    double y;
    double z;
    while (std::cin >> id >> x >> y >> z) {
        points.emplace_back(x, y, z);
        ids.emplace(key_for(x, y, z), id);
    }

    if (points.size() < 4) {
        std::cerr << "need at least 4 points\n";
        return 2;
    }

    Mesh mesh;
    const auto start = std::chrono::steady_clock::now();
    CGAL::convex_hull_3(points.begin(), points.end(), mesh);
    const auto finish = std::chrono::steady_clock::now();

    std::vector<std::array<std::uint32_t, 3>> facets;
    facets.reserve(mesh.number_of_faces());

    for (const auto face : mesh.faces()) {
        std::array<std::uint32_t, 3> tri{};
        std::size_t n = 0;
        for (const auto vertex : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
            if (n >= 3) {
                std::cerr << "non-triangular hull facet\n";
                return 3;
            }
            const auto found = ids.find(key_for(mesh.point(vertex)));
            if (found == ids.end()) {
                std::cerr << "hull vertex did not match input coordinate\n";
                return 4;
            }
            tri[n++] = found->second;
        }
        if (n != 3) {
            std::cerr << "degenerate hull facet\n";
            return 5;
        }
        std::sort(tri.begin(), tri.end());
        facets.push_back(tri);
    }

    std::sort(facets.begin(), facets.end());
    facets.erase(std::unique(facets.begin(), facets.end()), facets.end());

    for (const auto &tri : facets) {
        std::cout << tri[0] << ' ' << tri[1] << ' ' << tri[2] << '\n';
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(finish - start).count();
    std::cerr << "CGAL-HULL3 points=" << points.size() << " facets=" << facets.size()
              << " build_ms=" << elapsed_ms << '\n';
    return 0;
}
