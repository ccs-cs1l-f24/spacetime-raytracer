// softbody particle
struct Particle {
    ivec4 immediate_neighbors; // left/up/right/down
    ivec4 diagonal_neighbors;  // tl/tr/bl/br
    vec2 ground_pos;
    vec2 ground_vel;
    float rest_mass;
    uint object_index;
    uint id; // MUST BE UNIQUE
    uint _a;
};

// gotta love uniform buffers automatically padding to 16 bytes :/
// we'll represent it explicitly here
struct Object {
    // add to (immediate/diagonal)_neighbors to get actual location in particle buf of said neighbors
    uint offset;
    uint material_index;
    uint _a;
    uint _b;
};

// 16 bytes
struct WorldlineVertex {
    vec3 pos;
    uint object_index;
};

// copied from sebastian lague's vid
// idk if it's actually a good hash function
// whatever it works for him
// anyways this hash will index start_indices
uint hash_key_from_cell(ivec2 coord, uint num_particles) {
    uint a = uint(abs(coord.x)) * 15823;
    uint b = uint(abs(coord.y)) * 9737333;
    return (a + b) % num_particles;
}
