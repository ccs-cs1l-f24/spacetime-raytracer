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

layout(set = 0, binding = 0) buffer Particles {
    // cell hash, particle index (sorted by cell hash)
    Particle particles[];
};

layout(set = 1, binding = 0) buffer CollisionGrid1 {
    // cell hash, particle index (sorted by cell hash)
    ivec2 spatial_lookup[];
};
layout(set = 1, binding = 1) buffer CollisionGrid2 {
    // cell hash => where associated particle indices start in spatial_lookup
    int start_indices[];
};

layout(push_constant) uniform Settings {
    // workgroup size is 256 so this is to make sure we don't overread/write the particle bufs
    uint num_particles;
    float grid_resolution;
    uint spatial_lookup_len;
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
    return hash % spatial_lookup_len;
}

#ifdef CGRID_HASH
    void main() {
        uint index = gl_GlobalInvocationID.x;
        Particle p = particles[index];
        ivec2 cell_coord = floor(p.ground_pos / grid_resolution);
        //uint cell_hash = hash()
    }
#endif

#ifdef CGRID_SORT
    void main() {
        uint index = gl_GlobalInvocationID.x;
    }
#endif
