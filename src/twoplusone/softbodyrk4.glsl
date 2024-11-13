#version 460

#pragma vscode_glsllint_stage : comp

// see softbody.rs
// (should be) the same memory layout
struct Particle {
    ivec4 immediate_neighbors;
    ivec4 diagonal_neighbors;
    vec2 ground_pos;
    vec2 ground_vel;
    float rest_mass;
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

// https://www.youtube.com/watch?v=rSKMYc1CQHE
// https://web.archive.org/web/20140725014123/https://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf
// i love sebastian lague :))
// layout(set = 1, binding = 0) buffer CollisionGrid {
//     // cell hash, particle index (sorted by cell hash)
//     ivec2 spatial_lookup[];
//     // cell hash => where associated particle indices start in spatial_lookup
//     int start_indices[];
// };

layout(push_constant) uniform DebugSettings {
    // rk4 timestep in cs/s
    // (0.01?)
    float h;
};

// the forces applied to a particle are:
// - springs
// - collisions
// - global forces (gravity?, wind?, etc)
vec2 get_forces() {
    Particle particle = state_particles[gl_GlobalInvocationID.x];
    return vec2(0.0, 0.0);
}

// euler (state_particles is original_particles)
// more like rk1 amirite
#ifdef EULER
    void main() {
        uint index = gl_GlobalInvocationID.x;
        vec2 forces = get_forces();
        out_particles[index].ground_vel += forces * h;
        out_particles[index].ground_pos += original_particles[index].ground_vel * h;
    }
#endif

// rk4
#ifdef RK4STAGE_0
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        vec2 forces = get_forces();
        force_acc[index] += forces;
        vec2 new_vel = original_particles[index].ground_vel + forces * h / 2.0;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h / 2.0;
    }
#endif
#ifdef RK4STAGE_1
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
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
        vec2 forces = get_forces();
        force_acc[index] += forces;
    }
#endif
#ifdef RK4STAGE_4
    void main() { // relies on original, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        vec2 forces = force_acc[index];
        out_particles[index].vel += forces * h / 6.0;
        out_particles[index].pos += original_particles[index].vel * h;
        force_acc[index] = vec2(0.0);
    }
#endif
