CUDA 软光栅 — Part A（同学 A：正方体旋转）

文件：
  include/rotate.h   接口
  src/rotate.cu      rotate_cubes_kernel / build_animation_kernel
  src/main.cu        单独 benchmark

编译运行：
  powershell -ExecutionPolicy Bypass -File run_rotate.ps1

或手动：
  cmake -S . -B build
  cmake --build build --config Release
  .\build\Release\rotate_part_a.exe

输出：results/rotate_benchmark.txt
