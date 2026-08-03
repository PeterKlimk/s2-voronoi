#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="$root/target/competitors/vortex-src"
build_dir="$root/target/competitors/vortex-make-t16"
commit=3d59c666d69dbfb3d72513de19b8aae7ce4a57e0

if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/philipclaude/vortex.git "$source_dir"
  git -C "$source_dir" checkout --detach "$commit"
elif [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]]; then
  echo "existing Vortex checkout is not at pinned commit $commit" >&2
  exit 1
fi
if git -C "$source_dir" apply --check \
  "$root/benchmarks/competitors/vortex.patch" 2>/dev/null; then
  git -C "$source_dir" apply "$root/benchmarks/competitors/vortex.patch"
elif ! git -C "$source_dir" apply --reverse --check \
  "$root/benchmarks/competitors/vortex.patch" 2>/dev/null; then
  echo "Vortex compatibility patch is neither cleanly applicable nor already applied" >&2
  exit 1
fi
cp "$root/benchmarks/competitors/vortex_sphere.cpp" "$source_dir/src/bench_vortex_sphere.cpp"

cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -Dnum_cores=16 \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build "$build_dir" --target bench_vortex_sphere
