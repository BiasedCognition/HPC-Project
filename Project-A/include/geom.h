#pragma once

#include "types.h"

void make_unit_cube_verts(float* out8x3);
Scene make_scene(int num_cubes, unsigned seed);
Vec3 scene_center(const Scene& scene);
Mat3 rotation_y(float radians);
Vec3 mat3_mul_vec(Mat3 m, Vec3 v);
