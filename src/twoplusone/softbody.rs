use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use smallvec::SmallVec;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{BufferCopy, CopyBufferInfo},
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
};

use crate::boilerplate::BaseGpuState;

// ok so we can't have rigid bodies in special relativity
// the easiest approach would be to only use point particles
// which makes sense if you're making like a space game
// but that obviates any need for a fancy spacetime raytracer
// and doesn't leverage nearly the range of special relativity weirdness
// that large bodies do
// so instead we're using soft bodies :D
//
// there are many ways to simulate softbodies but basically how we're doing it here
// is creating 2d voxel/particle meshes at pixel-ish/sub-lightframe resolutions
// where each vertex is a point mass and connected to its 0-8 neighbors by springs
// and also can be affected by external forces (collision, wind/gravity(/normal), whatever)
// forces are resolved with an rk4 compute shader
//
// what about collisions? TODO (sebastian lague approach)
//
// we'll assume the speed of sound = c for all materials cause otherwise things'd be extra bad
//
// yes this does mean everything with physics will act somewhere between cloth and jello
// it'll be delightful :D
//
// as to constructing an efficient mesh
// cull each particle contained by all its neighbors
// get a tangent by averaging the particle's "surface border" edges, then rotate by 90 degrees
// do a constrained delaunay triangulation with all the rest
// (constrain w/surviving edges to prevent concave shapes from convexifying)
// we'll also have point-based rendering, for debugging purposes

pub struct SoftbodyRegistry {
    pub bodies: Vec<SoftbodyModel>,
}

// 52 bytes, not too bad
#[derive(BufferContents, Debug)]
#[repr(C)]
pub struct Particle {
    // index into particles vec (-1 for no corresponding)
    pub immediate_neighbors: [i32; 4], // left/up/right/down
    pub diagonal_neighbors: [i32; 4],  // tl/tr/bl/br
    // in lightseconds; worldspace
    pub ground_pos: [f32; 2],
    pub ground_vel: [f32; 2],
    pub rest_mass: f32,
}

// i guess we don't need much per-object state
// since softbodies are just collections of point masses
// well eventually we'll need materials i guess
pub struct SoftbodyModel {
    pub staging_buffer: Subbuffer<[Particle]>,
    pub gpu_buffer: Subbuffer<[Particle]>,
}

impl SoftbodyModel {
    // BLOCKS on the upload
    fn upload(&self, particles: &[Particle], base: &BaseGpuState) {
        // cpu copy
        let staging_ptr = self.staging_buffer.mapped_slice().unwrap().as_ptr() as *mut Particle;
        unsafe {
            std::ptr::copy(particles.as_ptr(), staging_ptr, particles.len());
        }
        let _ = staging_ptr;

        // cpu-gpu upload
        use vulkano::sync::GpuFuture;
        let mut cbuf_builder = base.create_primary_command_buffer();
        cbuf_builder
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (particles.len() * size_of::<Particle>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.staging_buffer.as_bytes().clone(),
                    self.gpu_buffer.as_bytes().clone(),
                )
            })
            .unwrap();
        let cbuf = cbuf_builder.build().unwrap();
        vulkano::sync::now(base.device.clone())
            .then_execute(base.queue.clone(), cbuf)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
    // TODO also add a download
    // for cpuside delaunay triangulation
}

fn allocate_and_upload_model(particles: &[Particle], base: &BaseGpuState) -> SoftbodyModel {
    let staging_buffer = Buffer::new_slice::<Particle>(
        base.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        particles.len() as u64,
    )
    .unwrap();
    let gpu_buffer = Buffer::new_slice::<Particle>(
        base.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST
                | BufferUsage::VERTEX_BUFFER
                | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        particles.len() as u64,
    )
    .unwrap();
    let model = SoftbodyModel {
        staging_buffer,
        gpu_buffer,
    };
    model.upload(particles, base);
    model
}

// we want resolutions of approximately 1 lightstep (ch where h is simulation tick)
// let's set an arbitrary resolution of 0.005 cs per pixel (200 particles per lightsecond)
// please only feed this 8-bit depth RGB images cause everything else will fail
// is BLOCKING ON GPU ACTIONS
pub fn image_to_softbody<R: std::io::Read>(r: R, base: &BaseGpuState) -> SoftbodyModel {
    let decoder = png::Decoder::new(r);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let buf: Vec<_> = buf
        .chunks(3)
        .map(|items| (items[0], items[1], items[2]))
        .collect();
    let width = info.width as usize;
    // let height = info.height as usize;
    let mut particles = vec![];
    let mut particle_map: HashMap<(i32, i32), usize> = HashMap::new();
    for (index, items) in buf.iter().enumerate() {
        let pos = ((index % width) as i32, (index / width) as i32);
        if *items != (0, 0, 0) {
            particles.push(Particle {
                immediate_neighbors: [-1, -1, -1, -1],
                diagonal_neighbors: [-1, -1, -1, -1],
                ground_pos: [pos.0 as f32 * 0.005, pos.1 as f32 * 0.005],
                ground_vel: [0.0, 0.0],
                rest_mass: 0.0,
            });
            particle_map.insert(pos, particles.len() - 1);
        }
    }
    for (&(cx, cy), &index) in particle_map.iter() {
        if let Some(neighbor) = particle_map.get(&(cx - 1, cy)) {
            particles[index].immediate_neighbors[0] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx, cy - 1)) {
            particles[index].immediate_neighbors[1] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx + 1, cy)) {
            particles[index].immediate_neighbors[2] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx, cy + 1)) {
            particles[index].immediate_neighbors[3] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx - 1, cy - 1)) {
            particles[index].diagonal_neighbors[0] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx + 1, cy - 1)) {
            particles[index].diagonal_neighbors[1] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx - 1, cy + 1)) {
            particles[index].diagonal_neighbors[2] = *neighbor as i32;
        }
        if let Some(neighbor) = particle_map.get(&(cx + 1, cy + 1)) {
            particles[index].diagonal_neighbors[3] = *neighbor as i32;
        }
    }
    allocate_and_upload_model(&particles, base)
}

// pub struct CollisionGrid {
//     //
// }

// particles <==> rk4
// particles ===> worldspace mesh ===> ring buffer

pub struct SoftbodyComputePipelines {
    set_layout: Arc<DescriptorSetLayout>,
    // pipeline layout (same for each of these pipelines lol)
    pipeline_layout: Arc<PipelineLayout>,
    // euler
    euler: Arc<ComputePipeline>,
    // rk4
    rk4_0: Arc<ComputePipeline>,
    rk4_1: Arc<ComputePipeline>,
    rk4_2: Arc<ComputePipeline>,
    rk4_3: Arc<ComputePipeline>,
    rk4_4: Arc<ComputePipeline>,
}

pub fn create_softbody_compute_pipelines(base: &BaseGpuState) -> SoftbodyComputePipelines {
    let set_layout = DescriptorSetLayout::new(
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
                tree.insert(3, binding.clone());
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
                size: 4,
            }],
            set_layouts: vec![set_layout.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("EULER", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_euler",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let euler = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("RK4STAGE_0", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_0",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_0 = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("RK4STAGE_1", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_1",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_1 = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("RK4STAGE_2", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_2",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_2 = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("RK4STAGE_3", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_3",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_3 = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.add_macro_definition("RK4STAGE_4", None);
    let shader = base
        .shader_loader
        .compile_into_spirv(
            include_str!("softbodyrk4.glsl"),
            shaderc::ShaderKind::DefaultCompute,
            "softbodyrk4_4",
            "main",
            Some(&opts),
        )
        .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(shader.as_binary()),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_4 = ComputePipeline::new(
        base.device.clone(),
        None,
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                pipeline_layout.clone(),
            )
        },
    )
    .unwrap();

    SoftbodyComputePipelines {
        set_layout,
        pipeline_layout,
        euler,
        rk4_0,
        rk4_1,
        rk4_2,
        rk4_3,
        rk4_4,
    }
}
