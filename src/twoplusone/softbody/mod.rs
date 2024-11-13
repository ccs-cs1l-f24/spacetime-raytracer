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
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    sync::GpuFuture,
};
use vulkano::{
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::PipelineBindPoint,
};

use crate::boilerplate::BaseGpuState;

pub mod point_render_nr;

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

// TODO replace with SoftbodyState
// pub struct SoftbodyRegistry {
//     pub bodies: Vec<SoftbodyModel>,
// }

// 64 bytes since glsl auto-pads from 56 to 64
#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Particle {
    // index into particles vec (-1 for no corresponding)
    pub immediate_neighbors: [i32; 4], // left/up/right/down
    pub diagonal_neighbors: [i32; 4],  // tl/tr/bl/br
    // in lightseconds; worldspace
    pub ground_pos: [f32; 2],
    pub ground_vel: [f32; 2],
    pub rest_mass: f32,
    // which "object" is this particle part of
    pub object_index: u32,
    pub _a: u32,
    pub _b: u32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Object {
    pub offset: u32,
    pub material_index: u32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct PushConstants {
    h: f32,
    num_particles: u32,
}

// we want resolutions of approximately 1 lightstep (ch where h is simulation tick)
// let's set an arbitrary resolution of 0.005 cs per pixel (200 particles per lightsecond)
// please only feed this 8-bit depth RGB images cause everything else will fail
// is BLOCKING ON GPU ACTIONS
pub fn image_to_softbody<R: std::io::Read>(r: R, object_index: u32) -> Vec<Particle> {
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
                object_index,
                _a: 0,
                _b: 0,
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
    particles
}

// pub struct CollisionGrid {
//     //
// }

pub struct SoftbodyState {
    // read back after each rk4 + culling
    particles: Vec<Particle>,
    objects: Vec<Object>, // fixed at 8192, since it's a uniform buffer

    // euler/rk4
    particle_staging: Subbuffer<[Particle]>,
    particle_buf: Subbuffer<[Particle]>,
    particle_buf_intermediate1: Subbuffer<[Particle]>,
    particle_buf_intermediate2: Subbuffer<[Particle]>,
    forcesum: Subbuffer<[f32]>,

    euler_ds: Arc<PersistentDescriptorSet>,
    // rk4_0_ds: PersistentDescriptorSet,
    // rk4_1_ds: PersistentDescriptorSet,
    // rk4_2_ds: PersistentDescriptorSet,
    // rk4_3_ds: PersistentDescriptorSet,
    // rk4_4_ds: PersistentDescriptorSet,

    // objects
    object_staging: Subbuffer<[Object]>,
    object_buf: Subbuffer<[Object]>,

    object_ds: Arc<PersistentDescriptorSet>,
    // collision

    // culling/meshing/render
}

impl SoftbodyState {
    // (how much should be allocated)
    const MAX_OBJECTS: u64 = 8192; // 2^16 bytes
    const MAX_PARTICLES: u64 = 1 << 20;
    pub fn create(base: &BaseGpuState, pipelines: &SoftbodyComputePipelines) -> Self {
        let particle_staging = Buffer::new_slice::<Particle>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let particle_buf = Buffer::new_slice::<Particle>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let particle_buf_intermediate1 = Buffer::new_slice::<Particle>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let particle_buf_intermediate2 = Buffer::new_slice::<Particle>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let forcesum = Buffer::new_slice::<f32>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let euler_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.rk4_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, particle_buf.clone()),
                WriteDescriptorSet::buffer(1, particle_buf.clone()),
                WriteDescriptorSet::buffer(2, particle_buf.clone()),
                WriteDescriptorSet::buffer(3, forcesum.clone()),
            ],
            [],
        )
        .unwrap();
        let object_staging = Buffer::new_slice::<Object>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            Self::MAX_PARTICLES,
        )
        .unwrap();
        let object_buf = Buffer::new_slice::<Object>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            Self::MAX_OBJECTS,
        )
        .unwrap();
        let object_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.objects_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, object_buf.clone())],
            [],
        )
        .unwrap();
        Self {
            particles: vec![],
            objects: vec![],
            particle_staging,
            particle_buf,
            particle_buf_intermediate1,
            particle_buf_intermediate2,
            forcesum,
            euler_ds,
            object_staging,
            object_buf,
            object_ds,
        }
    }

    // BLOCKS on gpu upload
    pub fn push(&self, base: &BaseGpuState) {
        let particle_staging_ptr =
            self.particle_staging.mapped_slice().unwrap().as_ptr() as *mut Particle;
        let object_staging_ptr =
            self.object_staging.mapped_slice().unwrap().as_ptr() as *mut Object;
        unsafe {
            std::ptr::copy(
                self.particles.as_ptr(),
                particle_staging_ptr,
                self.particles.len(),
            );
            std::ptr::copy(
                self.objects.as_ptr(),
                object_staging_ptr,
                self.objects.len(),
            );
        }
        let _ = particle_staging_ptr;
        let _ = object_staging_ptr;

        let mut cbuf_builder = base.create_primary_command_buffer();
        cbuf_builder
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.particles.len() * size_of::<Particle>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.particle_staging.as_bytes().clone(),
                    self.particle_buf.as_bytes().clone(),
                )
            })
            .unwrap()
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.objects.len() * size_of::<Object>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.object_staging.as_bytes().clone(),
                    self.object_buf.as_bytes().clone(),
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

    // TODO maybe add a thing to then copy particle_buf to some other buffer
    // for both points_norel and the particle culler
    pub fn submit_per_frame_compute(
        &self,
        base: &BaseGpuState,
        pipelines: &SoftbodyComputePipelines,
    ) -> Box<dyn GpuFuture> {
        let mut cmd_buf = base.create_primary_command_buffer();
        self.dispatch_euler(pipelines, &mut cmd_buf);
        vulkano::sync::now(base.device.clone())
            .then_execute(base.queue.clone(), cmd_buf.build().unwrap())
            .unwrap()
            .boxed()
    }

    pub fn dispatch_euler(
        &self,
        pipelines: &SoftbodyComputePipelines,
        cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        cmd_buf
            .bind_pipeline_compute(pipelines.euler.clone())
            .unwrap()
            .push_constants(
                pipelines.pipeline_layout.clone(),
                0,
                PushConstants {
                    h: 0.01, // TODO
                    num_particles: self.particles.len() as u32,
                },
            )
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.pipeline_layout.clone(),
                0,
                vec![self.euler_ds.clone(), self.object_ds.clone()],
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap();
    }
    // pub fn dispatch_rk4(&self, pipelines: &SoftbodyComputePipelines) {
    //     todo!()
    // }
    // pub fn update_collision_grid(&self, pipelines: &SoftbodyComputePipelines) {
    //     todo!()
    // }
    // pub fn cull_meshes(&self, pipelines: &SoftbodyComputePipelines) {
    //     todo!()
    // }

    // there should be exactly as many particles in self.particles as in the gpu buffer
    // (no new particles are being created, hopefully)
    // BLOCKS on gpu download
    pub fn pull(&mut self, base: &BaseGpuState) {
        let mut cbuf_builder = base.create_primary_command_buffer();
        cbuf_builder
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.particles.len() * size_of::<Particle>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.particle_buf.as_bytes().clone(),
                    self.particle_staging.as_bytes().clone(),
                )
            })
            .unwrap()
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.objects.len() * size_of::<Object>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.object_buf.as_bytes().clone(),
                    self.object_staging.as_bytes().clone(),
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
        let particle_staging_ptr =
            self.particle_staging.mapped_slice().unwrap().as_ptr() as *const Particle;
        let object_staging_ptr =
            self.object_staging.mapped_slice().unwrap().as_ptr() as *const Object;
        unsafe {
            std::ptr::copy(
                particle_staging_ptr,
                self.particles.as_mut_ptr(),
                self.particles.len(),
            );
            std::ptr::copy(
                object_staging_ptr,
                self.objects.as_mut_ptr(),
                self.objects.len(),
            );
        }
        let _ = particle_staging_ptr;
        let _ = object_staging_ptr;
    }

    // note that this function does NOT invoke push
    pub fn add_particles(&mut self, other: &mut Vec<Particle>, object: Object) {
        self.particles.append(other);
        self.objects.push(object);
    }
}

pub struct SoftbodyComputePipelines {
    rk4_set_layout: Arc<DescriptorSetLayout>,
    // collision_grid_set_layout: Arc<DescriptorSetLayout>,
    objects_set_layout: Arc<DescriptorSetLayout>,

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
    // collision (w/another pipeline layout)
    // (TODO)
}

pub fn create_softbody_compute_pipelines(base: &BaseGpuState) -> SoftbodyComputePipelines {
    let rk4_set_layout = DescriptorSetLayout::new(
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
    let objects_set_layout = DescriptorSetLayout::new(
        base.device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: {
                let binding = DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                };
                let mut tree = BTreeMap::new();
                tree.insert(0, binding);
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
                size: 8,
            }],
            set_layouts: vec![rk4_set_layout.clone(), objects_set_layout.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_include_callback(super::include_callback);
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
    opts.set_include_callback(super::include_callback);
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
    opts.set_include_callback(super::include_callback);
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
    opts.set_include_callback(super::include_callback);
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
    opts.set_include_callback(super::include_callback);
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
    opts.set_include_callback(super::include_callback);
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
        rk4_set_layout,
        objects_set_layout,
        pipeline_layout,
        euler,
        rk4_0,
        rk4_1,
        rk4_2,
        rk4_3,
        rk4_4,
    }
}
