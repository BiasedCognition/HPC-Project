#pragma once

#include "types.h"

#include <vector>

// Part A: CPU baseline
void rotate_scene_serial(Scene& scene, float delta_y);
std::vector<Scene> build_animation_serial(const Scene& base, int frames, float step_rad);

// CPU: same vertex work as GPU kernels (for fair benchmark)
void rotate_vertices_serial(const Scene& scene, float* world_verts);
void build_animation_vertices_serial(const Scene& base, int frames, float step_rad, float* world_verts);

// Part A: GPU kernels (host wrappers)
void rotate_scene_gpu(const Scene& scene, float delta_y, float* d_world_verts);
std::vector<Scene> build_animation_gpu(const Scene& base, int frames, float step_rad);

// Benchmark (angle-only CPU vs GPU — unfair, kept for outline compatibility)
double benchmark_rotation_serial(const Scene& base, int frames, float step_rad, int repeats);
double benchmark_rotation_gpu(const Scene& base, int frames, float step_rad, int repeats);

// Fair benchmark: same vertex transform workload, GPU buffers pre-allocated
double benchmark_vertex_transform_serial(const Scene& scene, int repeats);
double benchmark_vertex_transform_gpu(const Scene& scene, int repeats);
double benchmark_animation_vertices_serial(const Scene& base, int frames, float step_rad, int repeats);
double benchmark_animation_vertices_gpu(const Scene& base, int frames, float step_rad, int repeats);
