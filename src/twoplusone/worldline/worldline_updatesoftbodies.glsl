#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"
#include "relativity.glsl"

// IMPORTANT ASSUMPTIONS THIS MESHING ALGORITHM MAKES:
// - particles never appear or disappear
// - particles that are connected at some time must have been connected at every past time
// if particles are connected in a "line" for which there is no defined in and out we render them as separate triangles
// this'd be so much easier if i could just use SDFs
// but noo the raytracing extensions only work with triangles

struct WorldlineVertex {
    vec3 pos; // (x, y, t)
    // (these are treated as flat in the vertex shaders)
    uint object_index; //
    uint particle_index;
    uint direction; // 0/1/2/3 is left/right/top/bottom
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer P1 {
    Particle prev_particles[];
};

layout(set = 0, binding = 1) readonly buffer P2 {
    Particle curr_particles[];
};

layout(set = 0, binding = 2) buffer SpatialLookup {
    // cell hash, particle index (sorted by cell hash)
    // one per particle
    uvec2 spatial_lookup[];
};
layout(set = 0, binding = 3) buffer StartIndices {
    // cell hash => where associated particle indices start in spatial_lookup
    // one per particle
    uint start_indices[];
};

layout(set = 1, binding = 0) writeonly buffer O1 {
    // for only this most recent layer of the model
    WorldlineVertex out_vertices[];
};

layout(set = 1, binding = 0) writeonly buffer O2 {
    // 0 starts at the prev_particles layer of worldline
    // num_particles starts at curr_particles
    uint out_indices[];
};

layout(push_constant) uniform Settings {
    uint num_particles;
    float grid_resolution;
    float radius;
    float epsilon;
};

// for each particle, sample 4 points in each cardinal direction radius+epsilon distance from its center
// if such a sampled point is not in the radius of any nearby particles *of the same object*
// then it's part of an object boundary :D
// connect each particle to the closest 2 particles *of the same object*

#ifdef IDENTIFY_BOUNDARY
    void main() {
        uint index = gl_GlobalInvocationID.x;

        Particle particle = curr_particles[index];

        // plot 4 points just outside the "area of effect" of this particle
        // in a little diamond around the particle
        vec2 p1 = particle.ground_pos - vec2(radius + epsilon, 0.0); // left
        vec2 p2 = particle.ground_pos + vec2(radius + epsilon, 0.0); // right
        vec2 p3 = particle.ground_pos - vec2(0.0, radius + epsilon); // top
        vec2 p4 = particle.ground_pos + vec2(0.0, radius + epsilon); // bottom

        bool i1, i2, i3, i4 = true;

        ivec2 cell_coord = ivec2(floor(particle.ground_pos / grid_resolution));
        for (int i = 0; i < 9; i++) { // 0 to 8 (inclusive) --- i=4 is 0,0
            index = start_indices[hash_key_from_cell(cell_coord + ivec2((i % 3) - 1, (i / 3) - 1), num_particles)];
            if (index == 4294967295) continue; // no particles at that grid cell
            do {
                Particle other = curr_particles[spatial_lookup[index++].y];
                // ignore yourself
                if (other.ground_pos == particle.ground_pos) continue;
                // ignore particles that aren't from your same object
                if (other.object_index != particle.object_index) continue;
                // are any of the 4 points contained in another particle's radius
                if (distance(p1, other.ground_pos) < radius) i1 = false;
                if (distance(p2, other.ground_pos) < radius) i2 = false;
                if (distance(p3, other.ground_pos) < radius) i3 = false;
                if (distance(p4, other.ground_pos) < radius) i4 = false;
            } while (index < num_particles && spatial_lookup[index].x == spatial_lookup[index + 1].x);
        }
    }
#endif

#ifdef BOUNDARY_CONNECT
    void main() {
        uint index = gl_GlobalInvocationID.x;
    }
#endif
