#version 460

#pragma vscode_glsllint_stage : comp

#include "common.glsl"
#include "relativity.glsl"

// IMPORTANT ASSUMPTIONS THIS MESHING ALGORITHM MAKES:
// - particles never appear or disappear
// - particles that are connected at some time must have been connected at every past time
// - there are no particles connected in a "line"
//   i.e. at least two of its neighbors are connected to each other
//   (particles w/no neighbors become triangles)

struct WorldlineVertex {
    vec3 pos; // (x, y, t)
    uint object_index;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer P1 {
    Particle prev_particles[];
};

layout(set = 0, binding = 1) readonly buffer P2 {
    Particle curr_particles[];
};

layout(set = 1, binding = 0) buffer Stuff {
    uint allocated_space[];
};

layout(set = 2, binding = 0) writeonly buffer O1 {
    // for only this most recent layer of the model
    WorldlineVertex out_vertices[];
};

layout(set = 2, binding = 0) writeonly buffer O2 {
    // 0 starts at the prev_particles layer of worldline
    // num_particles starts at curr_particles
    uint out_indices[];
};

layout(push_constant) uniform Settings {
    uint num_particles;
};

void main() {
    uint index = gl_GlobalInvocationID.x;

    Particle particle = curr_particles[index];
}
 