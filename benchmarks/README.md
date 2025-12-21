# mEngine Benchmarks

## Overview

There are currently two workflows to benchmark TensorRT-LLM:
* [`mengine-bench`](../docs/source/performance/perf-benchmarking.md)
  - `mengine-bench` is native to mEngine and is a Python benchmarker for reproducing and testing the performance of mEngine.
  - _NOTE_: This benchmarking suite is a current work in progress and is prone to large changes.
* [C++ benchmarks](./cpp)
  - The recommended workflow that uses mEngine C++ API and can take advantage of the latest features of mEngine.
