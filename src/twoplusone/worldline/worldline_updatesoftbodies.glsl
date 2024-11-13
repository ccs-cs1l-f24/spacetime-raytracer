#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"
#include "relativity.glsl"

// 32 bytes
struct WorldlineVertex {
    vec3 pos; // (x, y, t)
    // (these are treated as flat in the vertex shaders)
    uint object_index;
    // (particle.id << 4) | direction (1/2/4/8 is left/right/up/down)
    uint packed_id;
    // the packed_ids of the vertex's neighbors
    // (the diamond)
    // is -1 if they don't exist
    int neighbor_1_id;
    int neighbor_2_id;
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
    float radius; // shouldn't be greater than like 1.5 * immediate_neighbor_dist
};

// for each particle, sample 4 points in each cardinal direction radius distance from its center
// if such a sampled point is not in the radius of any nearby particles *that are connected to this particle*
// then it's part of an object boundary :D
// connect each particle to the closest 2 particles *that are connected to the particle*

uint hash_edge(WorldlineVertex v1, WorldlineVertex v2, uint num_vertices) {
    uint i1 = v1.packed_id;
    uint i2 = v2.packed_id;
    // sebastian lague's hash function strikes again lmao
    return (i1 * 15823) + (i2 * 9737333) % num_vertices;
}

#ifdef IDENTIFY_BOUNDARY
    void main() {
        uint index = gl_GlobalInvocationID.x;

        Particle particle = particles[index];

        // plot 4 points just outside the "area of effect" of this particle
        // in a little diamond around the particle
        vec2 p1 = particle.ground_pos - vec2(radius, 0.0); // left
        vec2 p2 = particle.ground_pos + vec2(radius, 0.0); // right
        vec2 p3 = particle.ground_pos - vec2(0.0, radius); // top
        vec2 p4 = particle.ground_pos + vec2(0.0, radius); // bottom

        bool i1, i2, i3, i4 = true;

        ivec2 cell_coord = ivec2(floor(particle.ground_pos / grid_resolution));
        for (int i = 0; i < 9; i++) { // 0 to 8 (inclusive) --- i=4 is 0,0
            index = start_indices[hash_key_from_cell(cell_coord + ivec2((i % 3) - 1, (i / 3) - 1), num_particles)];
            if (index == 4294967295) continue; // no particles at that grid cell
            do {
                Particle other = particles[spatial_lookup[index++].y];
                // ignore yourself
                if (other.ground_pos == particle.ground_pos) continue;
                // ignore particles that aren't the same object as you
                if (other.object_index != particle.object_index) continue;
                // are any of the 4 points contained in another particle's radius
                if (distance(p1, other.ground_pos) < radius) i1 = false;
                if (distance(p2, other.ground_pos) < radius) i2 = false;
                if (distance(p3, other.ground_pos) < radius) i3 = false;
                if (distance(p4, other.ground_pos) < radius) i4 = false;
            } while (index < num_particles && spatial_lookup[index].x == spatial_lookup[index + 1].x);
        }
    }
#endif
