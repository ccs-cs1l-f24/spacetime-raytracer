use std::sync::Arc;

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
    debugcfg: &crate::debugui::HotswapConfig,
    cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    softbodies: &super::SoftbodyState,
    point_render_pipelines: &PointRenderPipelines,
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
                    debugcfg.scale.recip() * aspect_ratio.recip(),
                    debugcfg.scale.recip(),
                ]
            } else {
                [
                    debugcfg.scale.recip(),
                    debugcfg.scale.recip() * aspect_ratio,
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

pub fn create_point_render_pipelines(base: &BaseGpuState) -> PointRenderPipelines {
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("VERTEX_SHADER", None);
    let vertex_shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("points_norel.glsl"),
            shaderc::ShaderKind::DefaultVertex,
            "points_norel_vert",
            "main",
            Some(&opts),
        )
        .unwrap();
    let vertex_shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(vertex_shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("FRAGMENT_SHADER", None);
    let fragment_shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("points_norel.glsl"),
            shaderc::ShaderKind::DefaultFragment,
            "points_norel_frag",
            "main",
            Some(&opts),
        )
        .unwrap();
    let fragment_shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(fragment_shader.as_binary()),
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
                size: 8,
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
                .attributes([(
                    0,
                    VertexInputAttributeDescription {
                        binding: 0,
                        format: Format::R32G32_SFLOAT,
                        offset: 32, // offset 32 bytes into super::softbody::Particle to get ground_pos
                    },
                )])
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
    let render_pipeline =
        GraphicsPipeline::new(base.device.clone(), None, create_info.clone()).unwrap();
    PointRenderPipelines {
        render_pipeline_layout,
        render_pipeline,
    }
}
