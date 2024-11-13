#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"
#include "relativity.glsl"

// n = 1/2/4/8 is left/right/up/down
#define PACK(p, n) int((p.id << 4) | n)

// 32 bytes
struct IntermediateSoftbodyWorldlineVertex {
    vec3 pos; // (x, y, t)
    uint object_index;
    // (particle.id << 4) | direction (1/2/4/8 is left/right/up/down)
    uint packed_id;
    // the packed_ids of the vertex's neighbors (spawned by the shared particle)
    // one or both may be -1 (for does not exist)
    int sibling_1_id;
    int sibling_2_id;
    // used to signal whether the vertex exists (-1 for DNE, 0 for exists)
    int flag;
};

// 32 bytes
struct EdgeEntry {
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

layout(set = 1, binding = 0) buffer Vertices {
    // there are 4 vertices allocated for each particle
    IntermediateSoftbodyWorldlineVertex vertices[];
};
layout(set = 1, binding = 1) buffer Edges {
    // there are 8 edges allocated for each particle
    EdgeEntry edges;
};

// there are 8 edges allocated for each particle (2 per generated vertex)
// the below arrays are indexed by the edge hashes of two `IntermediateSoftbodyWorldlineVertex`s
layout(set = 2, binding = 0) buffer EdgeMap1 {
    // -1 for tombstone, 0 for does not exist, 1 for exists;
    // reset each time the edges are regenerated
    int ledger[];
};
layout(set = 2, binding = 1) buffer EdgeMap2 {
    // data at edge_map[i] where ledger[i] != 1 is garbage
    EdgeEntry edge_map[];
};

layout(push_constant) uniform Settings {
    uint num_particles;
    float grid_resolution;
    float radius; // shouldn't be greater than like 1.5 * immediate_neighbor_dist
    float time; // (the z coord that should be used)
    uint edge_map_capacity; // should be a power of 2 (should be the closest power of 2 to num_particles * 8)
};

// for each particle, sample 4 points in each cardinal direction radius distance from its center
// if such a sampled point is not in the radius of any nearby particles *that are connected to this particle*
// then it's part of an object boundary :D
// connect each particle to the closest 2 particles *that are connected to the particle*
uint hash_edge(IntermediateSoftbodyWorldlineVertex v1, IntermediateSoftbodyWorldlineVertex v2) {
    uint i1 = v1.packed_id;
    uint i2 = v2.packed_id;
    // sebastian lague's hash function strikes again lmao
    return ((i1 * 15823u) + (i2 * 9737333u)) % edge_map_capacity;
}

// https://nosferalatu.com/SimpleGPUHashTable.html
void register_edge(IntermediateSoftbodyWorldlineVertex v1, IntermediateSoftbodyWorldlineVertex v2) {
    uint hash = hash_edge(v1, v2);
    // we use atomic to avoid sync problems when hash collisions happen
    while (atomicCompSwap(ledger[hash], 0, 1) != 0)
        hash = (hash + 1u) % edge_map_capacity;
    EdgeEntry entry;
    entry.v1 = WorldlineVertex(v1.pos, v1.object_index);
    entry.v2 = WorldlineVertex(v2.pos, v2.object_index);
    edge_map[hash] = entry;
}

#ifdef IDENTIFY_VERTICES
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

        IntermediateSoftbodyWorldlineVertex vtx;
        vtx.object_index = particle.object_index;
        if (i1) {
            vtx.pos = vec3(p1, time);
            vtx.packed_id = PACK(particle, 1);
            vtx.sibling_1_id = i3 ? PACK(particle, 4) : -1;
            vtx.sibling_2_id = i4 ? PACK(particle, 8) : -1;
            vtx.flag = 0;
        } else vtx.flag = -1;
        vertices[index * 4] = vtx;
        if (i2) {
            vtx.pos = vec3(p2, time);
            vtx.packed_id = PACK(particle, 2);
            vtx.sibling_1_id = i3 ? PACK(particle, 4) : -1;
            vtx.sibling_2_id = i4 ? PACK(particle, 8) : -1;
            vtx.flag = 0;
        } else vtx.flag = -1;
        vertices[index * 4 + 1] = vtx;
        if (i3) {
            vtx.pos = vec3(p3, time);
            vtx.packed_id = PACK(particle, 4);
            vtx.sibling_1_id = i1 ? PACK(particle, 1) : -1;
            vtx.sibling_2_id = i2 ? PACK(particle, 2) : -1;
            vtx.flag = 0;
        } else vtx.flag = -1;
        vertices[index * 4 + 2] = vtx;
        if (i4) {
            vtx.pos = vec3(p4, time);
            vtx.packed_id = PACK(particle, 8);
            vtx.sibling_1_id = i1 ? PACK(particle, 1) : -1;
            vtx.sibling_2_id = i2 ? PACK(particle, 2) : -1;
            vtx.flag = 0;
        } else vtx.flag = -1;
        vertices[index * 4 + 3] = vtx;
    }
#endif

#ifdef GENERATE_COMPARE_EDGES
    void main() {
        uint index = gl_GlobalInvocationID.x;

        IntermediateSoftbodyWorldlineVertex vtx = vertices[index];
        if (vtx.flag == -1) return;
        
        Particle particle = particles[index / 4];

        // if the vertex has both its siblings, just use those edges
        if (vtx.sibling_1_id != -1 && vtx.sibling_2_id != -1) {
            //
        }

        ivec2 cell_coord = ivec2(floor(particle.ground_pos / grid_resolution));
        uint p_idx;
        IntermediateSoftbodyWorldlineVertex closest_vtx;
        IntermediateSoftbodyWorldlineVertex second_closest_vtx;
        for (int i = 0; i < 9; i++) { // 0 to 8 (inclusive) --- i=4 is 0,0
            p_idx = start_indices[hash_key_from_cell(cell_coord + ivec2((i % 3) - 1, (i / 3) - 1), num_particles)];
            if (p_idx == 4294967295) continue; // no particles at that grid cell
            do {
                Particle other = particles[spatial_lookup[p_idx++].y];
                for (int i = 0; i < 4; i++) {
                    IntermediateSoftbodyWorldlineVertex other_vtx = vertices[p_idx * 4 + i];
                    if (other_vtx.flag == -1) continue;
                    // now we compare :D
                }
            } while (p_idx < num_particles && spatial_lookup[p_idx].x == spatial_lookup[p_idx + 1].x);
        }
    }
#endif

#ifdef COMPACT_EDGES_AND_VERTICES
    void main() {
        uint index = gl_GlobalInvocationID.x;
    }
#endif

#ifdef CLEAR_EDGE_MAP
#endif

#ifdef WRITE_REGISTER_EDGES
#endif
