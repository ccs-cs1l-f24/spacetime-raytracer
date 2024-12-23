#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// https://www.youtube.com/watch?v=rSKMYc1CQHE
// https://web.archive.org/web/20140725014123/https://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
// i love sebastian lague :))
// i think we only need to update the spatial lookup once per full rk4 invokation
// probably actually only once per couple frames honestly
// since we consider every particle from a neighboring grid cell when doing detection
layout(set = 0, binding = 0) buffer Particles {
    // cell hash, particle index (sorted by cell hash)
    // one per particle
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
    // workgroup size is 256 so this is to make sure we don't overread/write the particle bufs
    uint num_particles;
    float grid_resolution;

    // just for the sorting method
    uint group_width;
    uint group_height;
    uint step_index;
};

// ok actually literally everything here is copied from sebastian lague
// https://www.youtube.com/watch?v=rSKMYc1CQHE
// just copying this url everywhere to show my appreciation

// generates a cell key for each particle
// seeds the spatial lookup with (cell_key, particle_index) pairs
#ifdef FILL_LOOKUP
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        Particle p = particles[index];
        ivec2 cell_coord = ivec2(floor(p.ground_pos / grid_resolution));
        uint cell_key = hash_key_from_cell(cell_coord, num_particles);
        spatial_lookup[index] = uvec2(cell_key, index);
    }
#endif

#ifdef SORT_LOOKUP
    // bitonic merge sort :)
    void main() {
        uint index = gl_GlobalInvocationID.x;
        uint h = index & (group_width - 1);
        uint index_low = h + (group_height + 1) * (index / group_width);
        uint index_high = index_low + (step_index == 0 ? group_height - 2 * h : (group_height + 1) / 2);
        if (index_high >= num_particles) return;
        uvec2 low = spatial_lookup[index_low];
        uvec2 high = spatial_lookup[index_high];
        if (low.x > high.x) {
            spatial_lookup[index_low] = high;
            spatial_lookup[index_high] = low;
        }
    }
#endif

#ifdef UPDATE_START_INDICES_1
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        // resets all the start indices to integer max value
        // this so that grid hashes can be identified as corresponding to no particles
        start_indices[index] = 4294967295;
    }
#endif

#ifdef UPDATE_START_INDICES_2
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        uint key = spatial_lookup[index].x;
        if (index == 0) start_indices[key] = 0;
        else {
            uint prev_key = spatial_lookup[index - 1].x;
            if (key != prev_key) start_indices[key] = index;
        }
    }
#endif
