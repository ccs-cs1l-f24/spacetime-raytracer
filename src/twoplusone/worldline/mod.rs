use std::{borrow::Borrow, collections::BTreeMap, sync::Arc};

use vulkano::{
    buffer::BufferContents,
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
};

use crate::boilerplate::BaseGpuState;

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
pub struct SoftbodyWorldlines {
    //
}

pub struct UpdateSoftbodiesPushConstants {
    num_particles: u32,
    grid_resolution: f32,
    radius: f32,
    time: f32,
    edge_map_capacity: u32,
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

pub struct UpdateSoftbodiesState {
    //
}

pub struct UpdateSoftbodiesComputePipelines {
    particles_set_layout: Arc<DescriptorSetLayout>,
    intermediate_edges_set_layout: Arc<DescriptorSetLayout>,
    edge_map_set_layout: Arc<DescriptorSetLayout>,

    identify_vertices: Arc<ComputePipeline>,
    // identify_edges: Arc<ComputePipeline>,
    // compact_edges: Arc<ComputePipeline>,
    // clear_edge_map: Arc<ComputePipeline>,
    // write_edges_to_worldline: Arc<ComputePipeline>,
}

pub fn create_update_softbodies(base: &mut BaseGpuState) -> UpdateSoftbodiesComputePipelines {
    base.register_cache("identify_vertices");
    base.register_cache("identify_edges");
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
    let intermediate_edges_set_layout = DescriptorSetLayout::new(
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
                tree
            },
            ..Default::default()
        },
    )
    .unwrap();
    let edge_map_set_layout = DescriptorSetLayout::new(
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
            set_layouts: vec![
                particles_set_layout.clone(),
                intermediate_edges_set_layout.clone(),
                edge_map_set_layout.clone(),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/worldline_updatesoftbodies_IDENTIFY_VERTICES.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let identify_vertices = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("identify_vertices")),
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
        intermediate_edges_set_layout,
        edge_map_set_layout,
        identify_vertices,
    }
}
