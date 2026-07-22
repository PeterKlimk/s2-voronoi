# Local benchmark environment: Ryzen 5 3600 / WSL2

Captured 2026-07-23. This describes the current development environment, not necessarily the final
publication machine.

- CPU: AMD Ryzen 5 3600
- Topology: 1 socket, 6 physical cores, 12 logical CPUs, 2 threads/core
- Architecture: x86-64
- Cache: 192 KiB L1d, 192 KiB L1i, 3 MiB L2, 16 MiB L3 as reported by `lscpu`
- Environment: WSL2, Linux `6.6.87.2-microsoft-standard-WSL2`
- Memory visible to WSL at capture: 10 GiB RAM and 4 GiB swap
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`, LLVM 22.1.6
- Cargo: `cargo 1.97.0 (c980f4866 2026-06-30)`

Before each campaign record the Windows host power plan, WSL memory limit, current available
memory/swap use, CPU affinity, background load, and whether boost behavior was controlled. WSL
results should not be silently combined with native-Linux results.
