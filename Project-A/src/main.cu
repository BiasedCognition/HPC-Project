#include "geom.h"
#include "rotate.h"

#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void print_speedup(const char* label, double serial_ms, double parallel_ms) {
    std::printf("  %-40s %10.2f ms  (speedup %.2fx)\n", label, parallel_ms,
                serial_ms / parallel_ms);
}

static bool angles_match(const std::vector<Scene>& a, const std::vector<Scene>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t f = 0; f < a.size(); ++f) {
        if (a[f].cubes.size() != b[f].cubes.size()) {
            return false;
        }
        for (size_t c = 0; c < a[f].cubes.size(); ++c) {
            const float da = a[f].cubes[c].angle_y - b[f].cubes[c].angle_y;
            if (std::fabs(da) > 1e-5f) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int num_cubes = 10;
    const int bench_cubes = 5000;
    const int num_frames = 12;
    const unsigned seed = 24281005u;
    const float step_rad = static_cast<float>(M_PI) / 6.0f;
    const int rot_repeats = 500;
    const int fair_repeats = 200;

    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    const Scene base = make_scene(num_cubes, seed);
    const Scene bench_scene = make_scene(bench_cubes, seed);
    const Vec3 center = scene_center(base);

    std::printf("=== Part A: Cube rotation ===\n");
    std::printf("GPU: %s\n", prop.name);
    std::printf("demo cubes: %d, benchmark cubes: %d, frames: %d, seed: %u\n", num_cubes,
                bench_cubes, num_frames, seed);
    std::printf("scene center: (%.2f, %.2f, %.2f)\n\n", center.x, center.y, center.z);

    const std::vector<Scene> scenes_cpu = build_animation_serial(base, num_frames, step_rad);
    const std::vector<Scene> scenes_gpu = build_animation_gpu(base, num_frames, step_rad);
    std::printf("animation angles CPU vs GPU: %s\n\n",
                angles_match(scenes_cpu, scenes_gpu) ? "OK" : "MISMATCH");

    std::printf("=== [1] Unfair compare (why GPU looks slower) ===\n");
    std::printf("CPU only updates angle_y; GPU transforms all vertices + alloc each round.\n");
    const double ms_angle_cpu = benchmark_rotation_serial(base, num_frames, step_rad, rot_repeats);
    const double ms_full_gpu = benchmark_rotation_gpu(base, num_frames, step_rad, rot_repeats);
    std::printf("  %-40s %10.2f ms\n", "CPU angle update only", ms_angle_cpu);
    std::printf("  %-40s %10.2f ms\n", "GPU full path (with alloc)", ms_full_gpu);
    std::printf("  => Not the same work; cannot prove GPU is faster here.\n\n");

    std::printf("=== [2] Fair compare: single-frame vertex transform ===\n");
    std::printf("Same work: rotate %d cube vertices, %d repeats, GPU buffers reused.\n",
                bench_cubes, fair_repeats);
    const double ms_v_cpu = benchmark_vertex_transform_serial(bench_scene, fair_repeats);
    const double ms_v_gpu = benchmark_vertex_transform_gpu(bench_scene, fair_repeats);
    std::printf("  %-40s %10.2f ms\n", "CPU rotate_vertices_serial", ms_v_cpu);
    print_speedup("GPU rotate_cubes_kernel", ms_v_cpu, ms_v_gpu);

    std::printf("\n=== [3] Fair compare: %d-frame animation vertices ===\n", num_frames);
    std::printf("Same work: all frames x all vertices, %d repeats, GPU buffers reused.\n",
                fair_repeats);
    const double ms_a_cpu =
        benchmark_animation_vertices_serial(bench_scene, num_frames, step_rad, fair_repeats);
    const double ms_a_gpu =
        benchmark_animation_vertices_gpu(bench_scene, num_frames, step_rad, fair_repeats);
    std::printf("  %-40s %10.2f ms\n", "CPU build_animation_vertices_serial", ms_a_cpu);
    print_speedup("GPU build_animation_kernel", ms_a_cpu, ms_a_gpu);

    std::printf("\nKernel config:\n");
    std::printf("  rotate_cubes_kernel:     grid=(num_cubes*8+255)/256, block=256\n");
    std::printf("  build_animation_kernel:  grid=(num_frames, ceil(total/256)), block=256\n");

    return 0;
}
