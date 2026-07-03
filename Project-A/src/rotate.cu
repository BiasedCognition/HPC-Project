#include "rotate.h"

#include "cuda_check.h"
#include "geom.h"

#include <chrono>
#include <cmath>
#include <vector>

__device__ inline float3 mat3_mul_float3(const float m[3][3], float3 v) {
    return make_float3(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                       m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                       m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
}

__device__ Mat3 rotation_y_device(float radians) {
    const float c = cosf(radians);
    const float s = sinf(radians);
    Mat3 r{};
    r.m[0][0] = c;
    r.m[0][2] = s;
    r.m[1][1] = 1.0f;
    r.m[2][0] = -s;
    r.m[2][2] = c;
    return r;
}

// grid: (num_cubes * 8 + 255) / 256, block: 256
// one thread per (cube, local_vertex)
__global__ void rotate_cubes_kernel(const float* local_verts, float* world_verts,
                                    const float* positions, const float* angles, int num_cubes) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_cubes * VERTS_PER_CUBE;
    if (idx >= total) {
        return;
    }

    const int cube = idx / VERTS_PER_CUBE;
    const int vi = idx % VERTS_PER_CUBE;

    const Mat3 R = rotation_y_device(angles[cube]);
    const float3 local =
        make_float3(local_verts[vi * 3 + 0], local_verts[vi * 3 + 1], local_verts[vi * 3 + 2]);
    const float3 rotated = mat3_mul_float3(R.m, local);
    const float3 pos =
        make_float3(positions[cube * 3 + 0], positions[cube * 3 + 1], positions[cube * 3 + 2]);

    world_verts[idx * 3 + 0] = rotated.x + pos.x;
    world_verts[idx * 3 + 1] = rotated.y + pos.y;
    world_verts[idx * 3 + 2] = rotated.z + pos.z;
}

// grid.x = num_frames, grid.y = ceil(total/256), block = 256
// one block row per frame, threads cover all cube vertices
__global__ void build_animation_kernel(const float* local_verts, const float* base_positions,
                                       const float* base_angles, float step_rad, int num_cubes,
                                       int num_frames, float* out_world) {
    const int frame = blockIdx.x;
    if (frame >= num_frames) {
        return;
    }

    const int idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int total = num_cubes * VERTS_PER_CUBE;
    if (idx >= total) {
        return;
    }

    const int cube = idx / VERTS_PER_CUBE;
    const int vi = idx % VERTS_PER_CUBE;
    const float angle = base_angles[cube] + step_rad * static_cast<float>(frame);

    const Mat3 R = rotation_y_device(angle);
    const float3 local =
        make_float3(local_verts[vi * 3 + 0], local_verts[vi * 3 + 1], local_verts[vi * 3 + 2]);
    const float3 rotated = mat3_mul_float3(R.m, local);
    const float3 pos = make_float3(base_positions[cube * 3 + 0], base_positions[cube * 3 + 1],
                                   base_positions[cube * 3 + 2]);

    const int out_base = (frame * total + idx) * 3;
    out_world[out_base + 0] = rotated.x + pos.x;
    out_world[out_base + 1] = rotated.y + pos.y;
    out_world[out_base + 2] = rotated.z + pos.z;
}

void rotate_scene_serial(Scene& scene, float delta_y) {
    for (CubeInstance& cube : scene.cubes) {
        cube.angle_y += delta_y;
    }
}

void rotate_vertices_serial(const Scene& scene, float* world_verts) {
    float local[VERTS_PER_CUBE * 3];
    make_unit_cube_verts(local);

    const int n = static_cast<int>(scene.cubes.size());
    for (int c = 0; c < n; ++c) {
        const CubeInstance& cube = scene.cubes[static_cast<size_t>(c)];
        const Mat3 R = rotation_y(cube.angle_y);
        for (int vi = 0; vi < VERTS_PER_CUBE; ++vi) {
            const Vec3 lv = {local[vi * 3 + 0], local[vi * 3 + 1], local[vi * 3 + 2]};
            const Vec3 rotated = mat3_mul_vec(R, lv);
            const int idx = c * VERTS_PER_CUBE + vi;
            world_verts[idx * 3 + 0] = rotated.x + cube.position.x;
            world_verts[idx * 3 + 1] = rotated.y + cube.position.y;
            world_verts[idx * 3 + 2] = rotated.z + cube.position.z;
        }
    }
}

void build_animation_vertices_serial(const Scene& base, int frames, float step_rad, float* world_verts) {
    float local[VERTS_PER_CUBE * 3];
    make_unit_cube_verts(local);

    const int n = static_cast<int>(base.cubes.size());
    const int total = n * VERTS_PER_CUBE;
    for (int f = 0; f < frames; ++f) {
        for (int c = 0; c < n; ++c) {
            const CubeInstance& cube = base.cubes[static_cast<size_t>(c)];
            const float angle = cube.angle_y + step_rad * static_cast<float>(f);
            const Mat3 R = rotation_y(angle);
            for (int vi = 0; vi < VERTS_PER_CUBE; ++vi) {
                const Vec3 lv = {local[vi * 3 + 0], local[vi * 3 + 1], local[vi * 3 + 2]};
                const Vec3 rotated = mat3_mul_vec(R, lv);
                const int idx = f * total + c * VERTS_PER_CUBE + vi;
                world_verts[idx * 3 + 0] = rotated.x + cube.position.x;
                world_verts[idx * 3 + 1] = rotated.y + cube.position.y;
                world_verts[idx * 3 + 2] = rotated.z + cube.position.z;
            }
        }
    }
}

static float* upload_unit_cube_verts() {
    float h_local[VERTS_PER_CUBE * 3];
    make_unit_cube_verts(h_local);
    float* d_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_local, sizeof(h_local)));
    CUDA_CHECK(cudaMemcpy(d_local, h_local, sizeof(h_local), cudaMemcpyHostToDevice));
    return d_local;
}

void rotate_scene_gpu(const Scene& scene, float delta_y, float* d_world_verts) {
    const int n = static_cast<int>(scene.cubes.size());
    std::vector<float> positions(static_cast<size_t>(n) * 3);
    std::vector<float> angles(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[i * 3 + 0] = scene.cubes[static_cast<size_t>(i)].position.x;
        positions[i * 3 + 1] = scene.cubes[static_cast<size_t>(i)].position.y;
        positions[i * 3 + 2] = scene.cubes[static_cast<size_t>(i)].position.z;
        angles[static_cast<size_t>(i)] = scene.cubes[static_cast<size_t>(i)].angle_y + delta_y;
    }

    float *d_pos = nullptr, *d_ang = nullptr;
    float* d_local = upload_unit_cube_verts();
    CUDA_CHECK(cudaMalloc(&d_pos, positions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ang, angles.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang, angles.data(), angles.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int total = n * VERTS_PER_CUBE;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    rotate_cubes_kernel<<<grid, block>>>(d_local, d_world_verts, d_pos, d_ang, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_pos);
    cudaFree(d_ang);
    cudaFree(d_local);
}

std::vector<Scene> build_animation_serial(const Scene& base, int frames, float step_rad) {
    std::vector<Scene> out;
    out.reserve(static_cast<size_t>(frames));
    Scene cur = base;
    for (int f = 0; f < frames; ++f) {
        rotate_scene_serial(cur, step_rad);
        out.push_back(cur);
    }
    return out;
}

std::vector<Scene> build_animation_gpu(const Scene& base, int frames, float step_rad) {
    const int n = static_cast<int>(base.cubes.size());
    std::vector<float> positions(static_cast<size_t>(n) * 3);
    std::vector<float> angles(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[i * 3 + 0] = base.cubes[static_cast<size_t>(i)].position.x;
        positions[i * 3 + 1] = base.cubes[static_cast<size_t>(i)].position.y;
        positions[i * 3 + 2] = base.cubes[static_cast<size_t>(i)].position.z;
        angles[static_cast<size_t>(i)] = base.cubes[static_cast<size_t>(i)].angle_y;
    }

    float *d_pos = nullptr, *d_ang = nullptr, *d_local = nullptr, *d_world = nullptr;
    const int total = n * VERTS_PER_CUBE;
    CUDA_CHECK(cudaMalloc(&d_pos, positions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ang, angles.size() * sizeof(float)));
    d_local = upload_unit_cube_verts();
    CUDA_CHECK(cudaMalloc(&d_world, static_cast<size_t>(frames) * total * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang, angles.data(), angles.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    build_animation_kernel<<<dim3(frames, (total + 255) / 256), 256>>>(
        d_local, d_pos, d_ang, step_rad, n, frames, d_world);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_world(static_cast<size_t>(frames) * total * 3);
    CUDA_CHECK(cudaMemcpy(h_world.data(), d_world, h_world.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<Scene> out;
    out.reserve(static_cast<size_t>(frames));
    for (int f = 0; f < frames; ++f) {
        Scene s = base;
        for (int c = 0; c < n; ++c) {
            s.cubes[static_cast<size_t>(c)].angle_y =
                angles[static_cast<size_t>(c)] + step_rad * static_cast<float>(f + 1);
        }
        out.push_back(std::move(s));
    }

    cudaFree(d_pos);
    cudaFree(d_ang);
    cudaFree(d_local);
    cudaFree(d_world);
    return out;
}

template <typename Fn>
static double time_ms(Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double benchmark_rotation_serial(const Scene& base, int frames, float step_rad, int repeats) {
    return time_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            Scene cur = base;
            for (int f = 0; f < frames; ++f) {
                rotate_scene_serial(cur, step_rad);
            }
        }
    });
}

double benchmark_rotation_gpu(const Scene& base, int frames, float step_rad, int repeats) {
    return time_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            build_animation_gpu(base, frames, step_rad);
        }
    });
}

static void pack_scene_arrays(const Scene& scene, std::vector<float>& positions,
                              std::vector<float>& angles) {
    const int n = static_cast<int>(scene.cubes.size());
    positions.resize(static_cast<size_t>(n) * 3);
    angles.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[i * 3 + 0] = scene.cubes[static_cast<size_t>(i)].position.x;
        positions[i * 3 + 1] = scene.cubes[static_cast<size_t>(i)].position.y;
        positions[i * 3 + 2] = scene.cubes[static_cast<size_t>(i)].position.z;
        angles[static_cast<size_t>(i)] = scene.cubes[static_cast<size_t>(i)].angle_y;
    }
}

double benchmark_vertex_transform_serial(const Scene& scene, int repeats) {
    const int n = static_cast<int>(scene.cubes.size());
    const int total = n * VERTS_PER_CUBE;
    std::vector<float> world(static_cast<size_t>(total) * 3);

    return time_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            rotate_vertices_serial(scene, world.data());
        }
    });
}

double benchmark_vertex_transform_gpu(const Scene& scene, int repeats) {
    const int n = static_cast<int>(scene.cubes.size());
    const int total = n * VERTS_PER_CUBE;

    std::vector<float> positions, angles;
    pack_scene_arrays(scene, positions, angles);

    float *d_pos = nullptr, *d_ang = nullptr, *d_local = nullptr, *d_world = nullptr;
    d_local = upload_unit_cube_verts();
    CUDA_CHECK(cudaMalloc(&d_pos, positions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ang, angles.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_world, static_cast<size_t>(total) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang, angles.data(), angles.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (total + block - 1) / block;

    rotate_cubes_kernel<<<grid, block>>>(d_local, d_world, d_pos, d_ang, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        rotate_cubes_kernel<<<grid, block>>>(d_local, d_world, d_pos, d_ang, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos);
    cudaFree(d_ang);
    cudaFree(d_local);
    cudaFree(d_world);
    return static_cast<double>(ms);
}

double benchmark_animation_vertices_serial(const Scene& base, int frames, float step_rad,
                                           int repeats) {
    const int n = static_cast<int>(base.cubes.size());
    const int total = n * VERTS_PER_CUBE;
    std::vector<float> world(static_cast<size_t>(frames) * total * 3);

    return time_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            build_animation_vertices_serial(base, frames, step_rad, world.data());
        }
    });
}

double benchmark_animation_vertices_gpu(const Scene& base, int frames, float step_rad,
                                        int repeats) {
    const int n = static_cast<int>(base.cubes.size());
    const int total = n * VERTS_PER_CUBE;

    std::vector<float> positions, angles;
    pack_scene_arrays(base, positions, angles);

    float *d_pos = nullptr, *d_ang = nullptr, *d_local = nullptr, *d_world = nullptr;
    d_local = upload_unit_cube_verts();
    CUDA_CHECK(cudaMalloc(&d_pos, positions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ang, angles.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_world, static_cast<size_t>(frames) * total * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang, angles.data(), angles.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    build_animation_kernel<<<dim3(frames, (total + 255) / 256), 256>>>(
        d_local, d_pos, d_ang, step_rad, n, frames, d_world);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        build_animation_kernel<<<dim3(frames, (total + 255) / 256), 256>>>(
            d_local, d_pos, d_ang, step_rad, n, frames, d_world);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos);
    cudaFree(d_ang);
    cudaFree(d_local);
    cudaFree(d_world);
    return static_cast<double>(ms);
}
