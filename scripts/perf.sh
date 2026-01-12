perf record -F 199 -g -- ./target/profiled/bench_voronoi --no-preprocess -n 5 1m
perf report --stdio --percent-limit 0.5 --sort symbol,dso > perf.txt
perf report --stdio --call-graph fractal,0.5,caller --percent-limit 0.5 --sort symbol > perf-callgraph.txt
