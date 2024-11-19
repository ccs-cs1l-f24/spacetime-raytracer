use std::{borrow::Borrow, collections::BTreeMap, sync::Arc};

use vulkano::{
    buffer::BufferContents,
    descriptor_set::{
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
    sync::{GpuFuture, PipelineStage},
};

use crate::boilerplate::BaseGpuState;

use super::softbody::SoftbodyState;

// ALOOFBODIES
// =============

// // a 3d model, a triangle mesh shell wrapping around the spacetime
// // that is consumed by a SINGLE aloofbody
// pub struct AloofbodyWorldline {
//     //
// }

// SOFTBODIES
// =============

// a 3d model, a triangle mesh shell wrapping around the spacetime
// that is consumed by all of our softbodies
// yes, all of them
// we stuff the object index (w/material and optics info)
// into each vertex
// pub struct SoftbodyWorldlines {
//     //
// }

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct UpdateSoftbodiesPushConstants {
    num_particles: u32,
    grid_resolution: f32,
    time: f32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct IntermediateSoftbodyWorldlineVertex {
    pub ground_pos: [f32; 3],
    pub object_index: u32,
    pub packed_id: u32,
    pub sibling_1_id: i32,
    pub sibling_2_id: i32,
    pub flag: i32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct WorldlineVertex {
    pub pos: [f32; 3],
    pub object_index: u32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Edge {
    pub v1: WorldlineVertex,
    pub v2: WorldlineVertex,
}

pub struct UpdateSoftbodiesState {
    particles_ds: Arc<PersistentDescriptorSet>,
}

impl UpdateSoftbodiesState {
    pub fn create(
        base: &BaseGpuState,
        pipelines: &UpdateSoftbodiesComputePipelines,
        softbody_state: &SoftbodyState,
    ) -> Self {
        let particles_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.particles_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, softbody_state.particle_buf.clone()),
                WriteDescriptorSet::buffer(1, softbody_state.spatial_lookup.clone()),
                WriteDescriptorSet::buffer(2, softbody_state.start_indices.clone()),
            ],
            [],
        )
        .unwrap();
        Self { particles_ds }
    }

    pub fn submit_update_worldlines(
        &self,
        base: &BaseGpuState,
        pipelines: &UpdateSoftbodiesComputePipelines,
        num_particles: u32,
    ) -> Box<dyn GpuFuture> {
        let mut cmd_buf = base.create_primary_command_buffer();
        unsafe {
            cmd_buf
                .write_timestamp(
                    base.query_pool.clone(),
                    crate::querybank::TOP_OF_MESHGEN,
                    PipelineStage::TopOfPipe,
                )
                .unwrap();
        }
        cmd_buf
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.pipeline_layout.clone(),
                0,
                vec![self.particles_ds.clone()],
            )
            .unwrap()
            .bind_pipeline_compute(pipelines.identify_vertices_and_edges.clone())
            .unwrap()
            .push_constants(
                pipelines.pipeline_layout.clone(),
                0,
                UpdateSoftbodiesPushConstants {
                    num_particles,
                    grid_resolution: super::consts::GRID_RESOLUTION,
                    time: 0.0,
                },
            )
            .unwrap()
            .dispatch([num_particles.div_ceil(256), 1, 1])
            .unwrap();
        unsafe {
            cmd_buf
                .write_timestamp(
                    base.query_pool.clone(),
                    crate::querybank::BOTTOM_OF_MESHGEN,
                    PipelineStage::BottomOfPipe,
                )
                .unwrap();
        }
        vulkano::sync::now(base.device.clone())
            .then_execute(base.queue.clone(), cmd_buf.build().unwrap())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .boxed()
    }
}

pub struct UpdateSoftbodiesComputePipelines {
    particles_set_layout: Arc<DescriptorSetLayout>,

    pipeline_layout: Arc<PipelineLayout>,

    identify_vertices_and_edges: Arc<ComputePipeline>,
}

pub fn create_update_softbodies(base: &mut BaseGpuState) -> UpdateSoftbodiesComputePipelines {
    base.register_cache("identify_vertices_and_edges");
    let particles_set_layout = DescriptorSetLayout::new(
        base.device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: {
                let binding = DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                };
                let mut tree = BTreeMap::new();
                tree.insert(0, binding.clone());
                tree.insert(1, binding.clone());
                tree.insert(2, binding.clone());
                tree
            },
            ..Default::default()
        },
    )
    .unwrap();
    let pipeline_layout = PipelineLayout::new(
        base.device.clone(),
        PipelineLayoutCreateInfo {
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: size_of::<UpdateSoftbodiesPushConstants>() as u32,
            }],
            set_layouts: vec![particles_set_layout.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/worldline_updatesoftbodies_IDENTIFY_VERTICES_AND_EDGES.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let identify_vertices_and_edges = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("identify_vertices_and_edges")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    UpdateSoftbodiesComputePipelines {
        particles_set_layout,
        pipeline_layout,
        identify_vertices_and_edges,
    }
}
