#pragma once

#include <vector>

static const int VERTS_PER_CUBE = 8;

struct Vec3 {
    float x, y, z;
};

struct Mat3 {
    float m[3][3];
};

struct CubeInstance {
    Vec3 position;
    float angle_y;
    unsigned char r, g, b;
};

struct Scene {
    std::vector<CubeInstance> cubes;
};
