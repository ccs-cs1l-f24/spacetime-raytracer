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

struct Object {
    uint offset; // in the main particle buffers
    uint material_index;
    //float k; // spring constant
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// rk4 time integrator (stable):
// f0 = f(X, t)
// f1 = f(X + h*f0/2, t + h/2)
// f2 = f(X + h*f1/2, t + h/2)
// f3 = f(X + h*f2,   t + h)
// X(t + h) = X + (h/6)(f0 + 2f1 + 2f2 + f3)

// is X
layout(set = 0, binding = 0) readonly buffer P1 {
    Particle original_particles[];
};

// represents one of the intermediate states
// e.g. X or X + h*f0/2 or X + h*f1/2 or X + h*f2
// used to create the next stage
layout(set = 0, binding = 1) readonly buffer P2 {
    Particle state_particles[];
};

// where the next intermediate states are written to
// e.g. X + h*f0/2 or X + h*f1/2 or X + h*f2
// also is where the final result is written to/calculated from
layout(set = 0, binding = 2) writeonly buffer P3 {
    Particle out_particles[];
};

layout(set = 0, binding = 3) buffer ForcesAccum {
    vec2 force_acc[];
};

// // https://www.youtube.com/watch?v=rSKMYc1CQHE
// // https://web.archive.org/web/20140725014123/https://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
// // i love sebastian lague :))
// // i think we only need to update the spatial lookup once per full rk4 invokation
// // probably actually only once per couple frames honestly
// // since we consider every particle from a neighboring grid cell when doing detection
// // (updating this grid goes in a different glsl file)
// layout(set = 1, binding = 0) buffer CollisionGrid {
//     // cell hash, particle index (sorted by cell hash)
//     ivec2 spatial_lookup[];
//     // cell hash => where associated particle indices start in spatial_lookup
//     int start_indices[];
// };

layout(set = 1, binding = 0) uniform Objects {
    // max uniform buffer size is 65536
    // so we get 8192 different objects at max
    // which seems like plenty
    Object objects[8192];
};

layout(push_constant) uniform Settings {
    // // mouse pos (for debugging purposes)
    // vec2 mouse_pos;
    // workgroup size is 256 so this is to make sure we don't overread/write the particle bufs
    uint num_particles;
    // rk4 timestep in cs/s
    // (0.01?)
    float h;
    // spring constants
    float immediate_neighbor_dist;
    float diagonal_neighbor_dist;
    float k;
};

// the forces applied to a particle are:
// - springs
// - collisions
// - global forces (gravity?, wind?, etc)
vec2 get_forces() {
    Object o = objects[0];
    Particle particle = state_particles[gl_GlobalInvocationID.x];
    vec2 forces = vec2(0.0, 0.0); // we accumulate forces here

    // springs adhering the object together
    // ideally the spring grid is fine enough that each |d| is at most 1 lightframe
    // and we can treat the spring forces as instantaneously accurate
    // F = -k(|d| - r) * (d/|d|) where d = p_1 - p_2
    for (int i = 0; i < 4; i++) { // this loop should unroll... if there are performance issues i can manually unroll
        if (particle.immediate_neighbors[i] != -1) {
            Particle n = state_particles[particle.immediate_neighbors[i]];
            vec2 d = particle.ground_pos - n.ground_pos;
            forces += -k * (length(d) - immediate_neighbor_dist) * normalize(d);
        }
    }
    for (int i = 0; i < 4; i++) {
        if (particle.diagonal_neighbors[i] != -1) {
            Particle n = state_particles[particle.diagonal_neighbors[i]];
            vec2 d = particle.ground_pos - n.ground_pos;
            forces += -k * (length(d) - diagonal_neighbor_dist) * normalize(d);
        }
    }
    if (particle.ground_pos.x < 0.2) {
        forces += vec2(0.1, 0.0);
    }

    return forces;
}

// euler (state_particles is original_particles)
// more like rk1 amirite
#ifdef EULER
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        out_particles[index].ground_vel += forces * h;
        out_particles[index].ground_pos += original_particles[index].ground_vel * h;
    }
#endif

// rk4
#ifdef RK4STAGE_0
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] = forces;
        vec2 new_vel = original_particles[index].ground_vel + forces * h / 2.0;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h / 2.0;
    }
#endif
#ifdef RK4STAGE_1
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] += forces * 2.0;
        vec2 new_vel = original_particles[index].ground_vel + forces * h / 2.0;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h / 2.0;
    }
#endif
#ifdef RK4STAGE_2
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] += forces * 2.0;
        vec2 new_vel = original_particles[index].ground_vel + forces * h;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h;
    }
#endif
#ifdef RK4STAGE_3
    void main() { // relies on state, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] += forces;
    }
#endif
#ifdef RK4STAGE_4
    void main() { // relies on original, out, force_acc
        Object o = objects[0]; // to stop vulkano from complaining about an unused descset binding
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = force_acc[index];
        vec2 vel = original_particles[index].ground_vel + forces * h / 6.0;
        out_particles[index].ground_vel = vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + vel * h;
        force_acc[index] = vec2(0.0);
    }
#endif
