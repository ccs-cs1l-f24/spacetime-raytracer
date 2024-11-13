use nalgebra::Vector2;

use crate::boilerplate::BaseGpuState;

pub mod point_render_nr;

mod softbody;
mod utils;

pub struct PipelineManager {
    pub point_pipelines: point_render_nr::PointRenderPipelines,
    pub softbody_compute: softbody::SoftbodyComputePipelines,
}

pub fn create_pipeline_manager(base: &BaseGpuState) -> PipelineManager {
    PipelineManager {
        point_pipelines: point_render_nr::create_point_render_pipelines(base),
        softbody_compute: softbody::create_softbody_compute_pipelines(base),
    }
}

pub struct World {
    pub softbodies: softbody::SoftbodyRegistry,
    // aloofbodies (floating)
    // player observer (potentially tethered to a softbody or pointbody)
}

pub fn create_world(base: &BaseGpuState) -> World {
    World {
        softbodies: softbody::SoftbodyRegistry {
            bodies: vec![softbody::image_to_softbody(
                include_bytes!("../../softbodyimages/testimg.png").as_slice(),
                base,
            )],
        },
    }
}

pub struct Observer {
    pub ground_pos: Vector2<f32>,
    pub ground_vel: Vector2<f32>,
}
