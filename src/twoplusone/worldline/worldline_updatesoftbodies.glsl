#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"

// 32 bytes
struct Edge {
    WorldlineVertex v1;
    WorldlineVertex v2;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer P2 {
    Particle particles[];
};
layout(set = 0, binding = 1) buffer SpatialLookup {
    // cell hash, particle index (sorted by cell hash)
    // one per particle
    uvec2 spatial_lookup[];
};
layout(set = 0, binding = 2) buffer StartIndices {
    // cell hash => where associated particle indices start in spatial_lookup
    // one per particle
    uint start_indices[];
};

layout(push_constant) uniform Settings {
    uint num_particles;
    float grid_resolution;
    float time; // (the z coord that should be used)
};

#ifdef IDENTIFY_VERTICES_AND_EDGES
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (start_indices[index] == 4294967295) {
        return;
    }
    // unfortunately, because of hash collisions
    // we need to go through every particle in the spatial lookup
    // that gets mapped to by the cell hash we're looking at
    uint i = start_indices[index];
    // i'm going to presume we have no more than 4 hash collisions
    // if we have more then i'll crash out the program
    ivec2 coords[4];
    coords[0] = ivec2(2147483647);
    coords[1] = ivec2(2147483647);
    coords[2] = ivec2(2147483647);
    coords[3] = ivec2(2147483647);
    uint diff_tiles = 0;
    do {
        Particle p = particles[spatial_lookup[i++].y];
        ivec2 cell_coord = ivec2(floor(p.ground_pos / grid_resolution));
        bool bail = false;
        for (int i = 0; i < 4; i++) if (cell_coord == coords[i]) bail = true;
        if (bail) continue;
        coords[diff_tiles++] = cell_coord;
        // OK, now we can finally do things with this cell coord we've identified
        // namely, potentially placing a vertex/some edges
        // TODO
    } while (i < num_particles && spatial_lookup[i].x == spatial_lookup[i + 1].x);
}
#endif

