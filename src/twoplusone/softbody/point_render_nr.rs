use std::{borrow::Borrow, sync::Arc};

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    format::{ClearValue, Format},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::{
                VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
                VertexInputState,
            },
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
};

use crate::boilerplate::BaseGpuState;

pub fn render(
    fbuf_index: u32,
    aspect_ratio: f32,
    base: &BaseGpuState,
    cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    softbodies: &super::SoftbodyState,
    point_render_pipelines: &PointRenderPipelines,
    cam_pos: [f32; 2],
    cam_scale: f32,
) {
    cmd_buf
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(ClearValue::Float([1.0, 1.0, 1.0, 1.0]))],
                ..RenderPassBeginInfo::framebuffer(
                    base.rpass_manager.main_pass.framebuffers[fbuf_index as usize].clone(),
                )
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .unwrap()
        .set_viewport(
            0,
            [base.swapchain_manager.viewport.clone()]
                .into_iter()
                .collect(),
        )
        .unwrap()
        .bind_pipeline_graphics(point_render_pipelines.render_pipeline.clone())
        .unwrap()
        .push_constants(
            point_render_pipelines.render_pipeline_layout.clone(),
            0,
            if aspect_ratio > 1.0 {
                [
                    cam_scale.recip() * aspect_ratio.recip(),
                    cam_scale.recip(),
                    cam_pos[0],
                    cam_pos[1],
                ]
            } else {
                [
                    cam_scale.recip(),
                    cam_scale.recip() * aspect_ratio,
                    cam_pos[0],
                    cam_pos[1],
                ]
            },
        )
        .unwrap();
    cmd_buf
        .bind_vertex_buffers(0, softbodies.particle_buf.clone())
        .unwrap()
        .draw(softbodies.particles.len() as u32, 1, 0, 0)
        .unwrap();
    cmd_buf.end_render_pass(Default::default()).unwrap();
}

pub struct PointRenderPipelines {
    pub render_pipeline_layout: Arc<PipelineLayout>,
    pub render_pipeline: Arc<GraphicsPipeline>,
}

pub fn create_point_render_pipelines(base: &mut BaseGpuState) -> PointRenderPipelines {
    base.register_cache("point_render_nr");
    let vertex_shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/points_norel_VERTEX_SHADER.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let fragment_shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/points_norel_FRAGMENT_SHADER.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let render_pipeline_layout = PipelineLayout::new(
        base.device.clone(),
        PipelineLayoutCreateInfo {
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::VERTEX,
                offset: 0,
                size: 16,
            }],
            set_layouts: vec![],
            ..Default::default()
        },
    )
    .unwrap();
    let create_info = GraphicsPipelineCreateInfo {
        stages: [
            PipelineShaderStageCreateInfo::new(vertex_shader.clone()),
            PipelineShaderStageCreateInfo::new(fragment_shader.clone()),
        ]
        .into_iter()
        .collect(),
        vertex_input_state: Some(
            VertexInputState::new()
                .attributes([
                    (
                        0,
                        VertexInputAttributeDescription {
                            binding: 0,
                            format: Format::R32G32_SFLOAT,
                            offset: 32, // offset 32 bytes into super::softbody::Particle to get ground_pos
                        },
                    ),
                    (
                        1,
                        VertexInputAttributeDescription {
                            binding: 0,
                            format: Format::R32_UINT,
                            offset: 52, // offset 52 bytes to get object_index
                        },
                    ),
                ])
                .bindings([(
                    0,
                    VertexInputBindingDescription {
                        stride: size_of::<super::Particle>() as u32,
                        input_rate: VertexInputRate::Vertex,
                    },
                )]),
        ),
        input_assembly_state: Some(InputAssemblyState {
            topology: PrimitiveTopology::PointList,
            ..Default::default()
        }),
        viewport_state: Some(ViewportState::default()),
        rasterization_state: Some(RasterizationState {
            polygon_mode: PolygonMode::Fill,
            ..Default::default()
        }),
        multisample_state: Some(MultisampleState::default()),
        color_blend_state: Some(ColorBlendState::with_attachment_states(
            base.rpass_manager.main_pass.subpass.num_color_attachments(), // aka 1
            ColorBlendAttachmentState::default(),
        )),
        dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        subpass: Some(base.rpass_manager.main_pass.subpass.clone().into()),
        ..GraphicsPipelineCreateInfo::layout(render_pipeline_layout.clone())
    };
    let render_pipeline = GraphicsPipeline::new(
        base.device.clone(),
        Some(base.get_cache("point_render_nr")),
        create_info.clone(),
    )
    .unwrap();
    PointRenderPipelines {
        render_pipeline_layout,
        render_pipeline,
    }
}
