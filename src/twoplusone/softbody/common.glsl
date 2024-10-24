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

// copied from sebastian lague's vid
// idk if it's actually a good hash function
// whatever it works for him
uint hash_key_from_cell(ivec2 coord, uint num_particles) {
    uint a = uint(coord.x) * 15823;
    uint b = uint(coord.y) * 9737333;
    return (a + b) % num_particles;
}
