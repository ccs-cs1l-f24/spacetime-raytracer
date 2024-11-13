#version 460

#pragma vscode_glsllint_stage : comp

#include "common.glsl"
#include "relativity.glsl"

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

layout(set = 1, binding = 0) uniform Objects {
    Object objects[1024];
};

layout(set = 2, binding = 0) buffer CollisionGrid1 {
    // cell hash, particle index (sorted by cell hash)
    Particle _dontusethis[];
};
layout(set = 2, binding = 1) buffer CollisionGrid2 {
    // cell hash, particle index (sorted by cell hash)
    uvec2 spatial_lookup[];
};
layout(set = 2, binding = 2) buffer CollisionGrid3 {
    // cell hash => where associated particle indices start in spatial_lookup
    uint start_indices[];
};

layout(push_constant) uniform Settings {
    // // mouse pos (for debugging purposes)
    // vec2 mouse_pos;
    // workgroup size is 256 so this is to make sure we don't overread/write the particle bufs
    uint num_particles;
    // rk4 timestep in cs/s
    // should be ~diagonal_neighbor_dist
    // since we want c * h (= 1 * h) to be no less than diagonal_neighbor_dist
    // so that information can't propagate faster than c
    float h;
    // spring constants
    float immediate_neighbor_dist;
    float diagonal_neighbor_dist;
    float k;
    // grid info
    float grid_resolution;
    float collision_repulsion_coefficient;
    float collision_distance;
    // other stuff
    float bond_break_threshold;
};

// vec2 get_surface_normal(Particle particle) {
//     vec2 normal = vec2(0.0);
//     for (int i = 0; i < 4; i++) { // this loop should unroll... if there are performance issues i can manually unroll
//         if (particle.immediate_neighbors[i] != -1) {
//             Particle n = state_particles[particle.immediate_neighbors[i]];
//             normal += particle.ground_pos - n.ground_pos;
//         }
//     }
//     for (int i = 0; i < 4; i++) {
//         if (particle.diagonal_neighbors[i] != -1) {
//             Particle n = state_particles[particle.diagonal_neighbors[i]];
//             normal += particle.ground_pos - n.ground_pos;
//         }
//     }
//     return normalize(normal);
// }

// the forces applied to a particle are:
// - springs
// - collisions
// - global forces (gravity?, wind?, etc)
vec2 get_forces() {
    Particle particle = state_particles[gl_GlobalInvocationID.x];
    Object obj = objects[particle.object_index];

    vec2 forces = vec2(0.0, 0.0); // we accumulate forces here

    // particle-particle collisions
    ivec2 cell_coord = ivec2(floor(particle.ground_pos / grid_resolution));
    uint index;
    for (int i = 0; i < 9; i++) { // 0 to 8 (inclusive) --- i=4 is 0,0
        index = start_indices[hash_key_from_cell(cell_coord + ivec2((i % 3) - 1, (i / 3) - 1), num_particles)];
        if (index == 4294967295) continue; // no particles at that grid cell
        do {
            // // no colliding when you're fully inside the softbody!
            // // NEVERMIND we actually want that
            // if (particle.immediate_neighbors[0] != -1 &&
            //     particle.immediate_neighbors[1] != -1 &&
            //     particle.immediate_neighbors[2] != -1 &&
            //     particle.immediate_neighbors[3] != -1 && 
            //     particle.diagonal_neighbors[0] != -1 &&
            //     particle.diagonal_neighbors[1] != -1 &&
            //     particle.diagonal_neighbors[2] != -1 &&
            //     particle.diagonal_neighbors[3] != -1) break;
            Particle p2 = state_particles[spatial_lookup[index++].y];
            // no colliding with yourself!
            if (p2.ground_pos == particle.ground_pos) continue;
            // no colliding with your neighbors!
            if (particle.immediate_neighbors[0] == index - 1 ||
                particle.immediate_neighbors[1] == index - 1 ||
                particle.immediate_neighbors[2] == index - 1 ||
                particle.immediate_neighbors[3] == index - 1 || 
                particle.diagonal_neighbors[0] == index - 1 ||
                particle.diagonal_neighbors[1] == index - 1 ||
                particle.diagonal_neighbors[2] == index - 1 ||
                particle.diagonal_neighbors[3] == index - 1) continue;
            vec2 d = particle.ground_pos - p2.ground_pos;
            if (length(d) < collision_distance) {
                forces += normalize(d) * collision_repulsion_coefficient;
            }
        } while (index < num_particles && spatial_lookup[index].x == spatial_lookup[index + 1].x);
    }

    // springs adhering the object together
    // ideally the spring grid is fine enough that each |d| is at most 1 lightframe
    // and we can treat the spring forces as instantaneously accurate
    // F = -k(|d| - r) * (d/|d|) where d = p_1 - p_2
    // despite being very simple they seem to produce the best effect!
    for (int i = 0; i < 4; i++) { // this loop should unroll... if there are performance issues i can manually unroll
        if (particle.immediate_neighbors[i] != -1) {
            Particle n = state_particles[particle.immediate_neighbors[i] + obj.offset];
            vec2 d = particle.ground_pos - n.ground_pos;
            forces += -k * (length(d) - immediate_neighbor_dist) * normalize(d);
            // // too stiff/not working
            // // "two coulomb potentials stacked on top of each other" - parsa
            // forces += normalize(d) * 1/(length(d)*length(d)) - 20000/((length(d) + 6)*(length(d) + 6));
            // // yukawa potential (not the right thing)
            // k^2 * (e^(-length(d)/immediate_neighbor_dist) * (r/immediate_neighbor_dist) + 1) / length(d)^2
            // forces += normalize(d) * k*k * (exp(-length(d)/immediate_neighbor_dist) * (length(d)/immediate_neighbor_dist + 1))/max(length(d) * length(d), 0.1);
        }
    }
    for (int i = 0; i < 4; i++) {
        if (particle.diagonal_neighbors[i] != -1) {
            Particle n = state_particles[particle.diagonal_neighbors[i] + obj.offset];
            vec2 d = particle.ground_pos - n.ground_pos;
            forces += -k * (length(d) - diagonal_neighbor_dist) * normalize(d);
        }
    }

    return forces;
}

// the bonds get broken in rk4_4 but only for particles
// particles_intermediate1 and particles_intermediate2 also need their bonds updated
// so we call this in rk4_0 and rk4_1 to propagate the breaking
void propagate_breaking(uint index) {
    out_particles[index].immediate_neighbors = original_particles[index].immediate_neighbors;
    out_particles[index].diagonal_neighbors = original_particles[index].diagonal_neighbors;
}

// euler (state_particles is original_particles)
// more like rk1 amirite
#ifdef EULER
    void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        vec2 acc = r_acc(forces, original_particles[index].ground_vel, original_particles[index].rest_mass);
        out_particles[index].ground_vel += acc * h;
        out_particles[index].ground_pos += original_particles[index].ground_vel * h;
        propagate_breaking(index);
    }
#endif

// rk4
#ifdef RK4STAGE_0
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] = forces;
        vec2 acc = r_acc(forces, original_particles[index].ground_vel, original_particles[index].rest_mass);
        vec2 new_vel = original_particles[index].ground_vel + acc * h / 2.0;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h / 2.0;
        propagate_breaking(index);
    }
#endif
#ifdef RK4STAGE_1
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] += forces * 2.0;
        vec2 acc = r_acc(forces, original_particles[index].ground_vel, original_particles[index].rest_mass);
        vec2 new_vel = original_particles[index].ground_vel + acc * h / 2.0;
        out_particles[index].ground_vel = new_vel;
        out_particles[index].ground_pos = original_particles[index].ground_pos + new_vel * h / 2.0;
        propagate_breaking(index);
    }
#endif
#ifdef RK4STAGE_2
    void main() { // relies on original, state, out, force_acc
        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = get_forces();
        force_acc[index] += forces * 2.0;
        vec2 acc = r_acc(forces, original_particles[index].ground_vel, original_particles[index].rest_mass);
        vec2 new_vel = original_particles[index].ground_vel + acc * h;
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
        // to stop vulkano from complaining about an unused descset binding
        Object o = objects[0];
        uvec2 u = spatial_lookup[0];

        uint index = gl_GlobalInvocationID.x;
        if (index >= num_particles) return;
        vec2 forces = force_acc[index];
        vec2 acc = r_acc(forces, original_particles[index].ground_vel, original_particles[index].rest_mass);
        vec2 vel = original_particles[index].ground_vel + acc * h / 6.0;
        // since timestamps aren't infinitely granular, acceleration can cause the velocity to be over c
        // we correct for that here :D
        if (length(vel) >= 1.0) vel = normalize(vel) * 0.9999;
        out_particles[index].ground_vel = vel;
        vec2 new_pos = original_particles[index].ground_pos + vel * h;
        out_particles[index].ground_pos = new_pos;
        force_acc[index] = vec2(0.0);

        // break bonds between particles that are too far apart
        Particle p = original_particles[index];
        Object obj = objects[p.object_index];
        for (int i = 0; i < 4; i++) {
            if (p.immediate_neighbors[i] != -1) {
                vec2 d = p.ground_pos - original_particles[p.immediate_neighbors[i] + obj.offset].ground_pos;
                if (length(d) > bond_break_threshold) {
                    out_particles[index].immediate_neighbors[i] = -1;
                    int j = i > 1 ? i - 2 : i + 2;
                    out_particles[p.immediate_neighbors[i] + obj.offset].immediate_neighbors[j] = -1;
                }
            }
            if (p.diagonal_neighbors[i] != -1) {
                vec2 d = p.ground_pos - original_particles[p.diagonal_neighbors[i] + obj.offset].ground_pos;
                if (length(d) > bond_break_threshold) {
                    out_particles[index].diagonal_neighbors[i] = -1;
                    int j = 3 - i;
                    out_particles[p.diagonal_neighbors[i] + obj.offset].diagonal_neighbors[j] = -1;
                }
            }
        }
    }
#endif
