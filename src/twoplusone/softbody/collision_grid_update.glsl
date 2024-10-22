#version 460

#pragma vscode_glsllint_stage : comp

struct Particle {
    ivec4 immediate_neighbors;
    ivec4 diagonal_neighbors;
    vec2 ground_pos;
    vec2 ground_vel;
    float rest_mass;
    uint object_index;
    uint _a; // we love padding :)
    uint _b;
};

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

// copied from sebastian lague's vid
// idk if it's actually a good hash function
// whatever it works for him
uint hash_cell(ivec2 coord) {
    uint a = uint(coord.x) * 15823;
    uint b = uint(coord.y) * 9737333;
    return a + b;
}

uint key_from_hash(uint hash) {
    return hash % num_particles;
}

// ok actually literally everything here is copied from sebastian lague

// generates a cell key for each particle
// seeds the spatial lookup with (cell_key, particle_index) pairs
#ifdef FILL_LOOKUP
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        Particle p = particles[index];
        ivec2 cell_coord = ivec2(floor(p.ground_pos / grid_resolution));
        uint cell_hash = hash_cell(cell_coord);
        uint cell_key = key_from_hash(cell_hash);
        spatial_lookup[index] = uvec2(cell_key, index);
    }
#endif

// sorts the spatial lookup
// writes the start indices
#ifdef SORT_LOOKUP
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

#ifdef UPDATE_START_INDICES
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        uint key = spatial_lookup[index].x;
        uint prev_key = index == 0 ? 4294967295 : spatial_lookup[index - 1].x;
        if (key != prev_key) start_indices[key] = index;
        else start_indices[key] = 4294967295;
    }
#endif
