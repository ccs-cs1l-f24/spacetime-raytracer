use vulkano::buffer::BufferContents;

// a 3d model, a triangle mesh shell wrapping around the spacetime
// that is consumed by all of our softbodies
// yes, all of them
// we stuff the object index (w/material and optics info)
// into each vertex
pub struct SoftbodyWorldlines {
    //
}

pub struct UpdateSoftbodiesPushConstants {
    num_particles: u32,
    grid_resolution: f32,
    radius: f32,
    epsilon: f32,
}

// a 3d model, a triangle mesh shell wrapping around the spacetime
// that is consumed by a SINGLE aloofbody
pub struct AloofbodyWorldline {
    //
}

//

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct WorldlineVertex {
    pub ground_pos: [f32; 3],
    pub object_index: u32,
}
