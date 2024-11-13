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
    uint object_index; // (treated as flat in the vertex shaders)
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer P1 {
    Particle prev_particles[];
};

layout(set = 0, binding = 1) readonly buffer P2 {
    Particle curr_particles[];
};

layout(set = 1, binding = 0) buffer Allocation {
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

// // if none of a particle's neighbors are connected to each other, we have a line
// // which is to say, no way to define an inside and outside for this particle
// bool is_line() {
//     if (p.immediate_neighbors[0] != -1 && p.diagonal_neighbors[0] != -1) { // left to top left
//         Particle left = original_particles[p.immediate_neighbors[0]];
//         if (left.immediate_neighbors[1] == p.diagonal_neighbors[0])
//             return false;
//     }
//     if (p.diagonal_neighbors[0] != -1 && p.immediate_neighbors[1] != -1) { // top left to top
//         Particle tl = original_particles[p.diagonal_neighbors[0]];
//         if (tl.immediate_neighbors[2] == p.immediate_neighbors[1])
//             return false;
//     }
//     if (p.immediate_neighbors[1] != -1 && p.diagonal_neighbors[1] != -1) { // top to top right
//         Particle top = original_particles[p.immediate_neighbors[1]];
//         if (top.immediate_neighbors[2] == p.diagonal_neighbors[1])
//             return false;
//      }
//     if (p.diagonal_neighbors[1] != -1 && p.immediate_neighbors[2] != -1) { // top right to right
//         Particle tr = original_particles[p.diagonal_neighbors[1]];
//         if (tr.immediate_neighbors[3] == p.immediate_neighbors[2])
//             return false;
//     }
//     if (p.immediate_neighbors[2] != -1 && p.diagonal_neighbors[3] != -1) { // right to bottom right
//         Particle right = original_particles[p.immediate_neighbors[2]];
//         if (right.immediate_neighbors[3] == p.diagonal_neighbors[3])
//             return false;
//     }
//     if (p.diagonal_neighbors[3] != -1 && p.immediate_neighbors[3] != -1) { // bottom right to bottom
//         Particle br = original_particles[p.diagonal_neighbors[3]];
//         if (br.immediate_neighbors[0] == p.immediate_neighbors[3])
//             return false;
//     }
//     if (p.immediate_neighbors[3] != -1 && p.diagonal_neighbors[2] != -1) { // bottom to bottom left
//         Particle bottom = original_particles[p.immediate_neighbors[3]];
//         if (bottom.immediate_neighbors[0] == p.diagonal_neighbors[2])
//             return false;
//     }
//     if (p.diagonal_neighbors[2] != -1 && p.immediate_neighbors[0] != -1) { // bottom left to left
//         Particle br = original_particles[p.diagonal_neighbors[2]];
//         if (br.immediate_neighbors[1] == p.immediate_neighbors[0])
//             return false;
//     }
//     return true;
// }

// TODO
// check the 4 potential triangles adjacent each edge considered
// if at least one on each side no need to render the edge
// if one on one side none on the other then render the edge
// if neither then don't render the edge but flag that one or more of the particles involved may need to be rendered special case-y

// TODO alternative approach
// go on a per-occupied-grid-square basis
// SDF all the particles nearby this one
// if the SDF doesn't cover this entire grid square
// then add the grid square to the mesh?

void main() {
    uint index = gl_GlobalInvocationID.x;

    Particle particle = curr_particles[index];

    if (particle.immediate_neighbors[0] != -1 &&
        particle.immediate_neighbors[1] != -1 &&
        particle.immediate_neighbors[2] != -1 &&
        particle.immediate_neighbors[3] != -1 && 
        particle.diagonal_neighbors[0] != -1 &&
        particle.diagonal_neighbors[1] != -1 &&
        particle.diagonal_neighbors[2] != -1 &&
        particle.diagonal_neighbors[3] != -1
    ) {
        // if you're in the middle of an object, no need to do anything
    }
}
 