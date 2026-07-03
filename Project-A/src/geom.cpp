#include "geom.h"

#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float rand01() { return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX); }
static float rand_range(float lo, float hi) { return lo + (hi - lo) * rand01(); }

static Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

void make_unit_cube_verts(float* out8x3) {
    const float v[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
    };
    for (int i = 0; i < 8; ++i) {
        out8x3[i * 3 + 0] = v[i][0];
        out8x3[i * 3 + 1] = v[i][1];
        out8x3[i * 3 + 2] = v[i][2];
    }
}

Scene make_scene(int num_cubes, unsigned seed) {
    std::srand(static_cast<int>(seed));
    Scene scene;
    scene.cubes.reserve(static_cast<size_t>(num_cubes));
    for (int i = 0; i < num_cubes; ++i) {
        CubeInstance c{};
        c.position = {rand_range(-2.0f, 2.0f), rand_range(-1.0f, 1.0f), rand_range(-2.0f, 2.0f)};
        c.angle_y = rand_range(0.0f, static_cast<float>(2.0 * M_PI));
        c.r = static_cast<unsigned char>(80 + std::rand() % 175);
        c.g = static_cast<unsigned char>(80 + std::rand() % 175);
        c.b = static_cast<unsigned char>(80 + std::rand() % 175);
        scene.cubes.push_back(c);
    }
    return scene;
}

Vec3 scene_center(const Scene& scene) {
    Vec3 c{0, 0, 0};
    if (scene.cubes.empty()) {
        return c;
    }
    for (const CubeInstance& cube : scene.cubes) {
        c = c + cube.position;
    }
    return c * (1.0f / static_cast<float>(scene.cubes.size()));
}

Mat3 rotation_y(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 r{};
    r.m[0][0] = c;
    r.m[0][2] = s;
    r.m[1][1] = 1.0f;
    r.m[2][0] = -s;
    r.m[2][2] = c;
    return r;
}

Vec3 mat3_mul_vec(Mat3 m, Vec3 v) {
    return {m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
            m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
            m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z};
}
