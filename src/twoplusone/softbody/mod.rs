use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
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
    sync::PipelineStage,
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
// so instead we're using softbodies :D
//
// there are many ways to simulate softbodies but basically how we're doing it here
// is creating 2d voxel/particle meshes at pixel-ish/sub-lightframe resolutions
// where each vertex is a point mass and connected to its 0-8 neighbors by springs
// and also can be affected by external forces (collision, wind/gravity(/normal), whatever)
// forces are resolved with an rk4 compute shader
//
// what about collisions? use the sebastian lague approach
//
// we'll assume the speed of sound = c for all materials cause otherwise things'd be extra bad
//
// yes this does mean everything with physics will act somewhere between cloth and jello
// it'll be delightful :D
//
// as to meshing... refer to the twoplusone/worldline module for that it's a whole other thing

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
    // UNIQUE
    pub id: u32,
    pub _a: u32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
// need to keep the size of this struct a multiple of 16!!!
// cause glsl uniform buffers are stupid >:(
pub struct Object {
    pub offset: u32,
    pub material_index: u32,
    pub _a: u32,
    pub _b: u32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Rk4PushConstants {
    pub(super) num_particles: u32,
    pub(super) h: f32,
    pub(super) immediate_neighbor_dist: f32,
    pub(super) diagonal_neighbor_dist: f32,
    pub(super) k: f32,
    pub(super) grid_resolution: f32,
    pub(super) collision_repulsion_coefficient: f32,
    pub(super) collision_distance: f32,
    pub(super) bond_break_threshold: f32,
}

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct CollisionGridPushConstants {
    num_particles: u32,
    grid_resolution: f32,
    // parameters for bitonic merge sort
    group_width: u32,
    group_height: u32,
    step_index: u32,
}

static MAX_PARTICLE_ID: AtomicU32 = AtomicU32::new(0);

// we want resolutions of approximately 1 lightstep (ch where h is simulation tick)
// see super::consts (consts mod in src/twoplusone/mod.rs) for more on that
// please only feed this 8-bit depth RGB images cause everything else will fail
// is BLOCKING ON GPU ACTIONS
pub fn image_to_softbody<R: std::io::Read>(
    r: R,
    object_index: u32,
    ground_pos_offset: [f32; 2],
    // TODO: should grid resolution be adjusted to be uniform in object frame rather than ground frame?
    // i'm thinking no
    starting_ground_vel: [f32; 2],
) -> Vec<Particle> {
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
                ground_pos: [
                    pos.0 as f32 * super::consts::IMMEDIATE_NEIGHBOR_DIST + ground_pos_offset[0],
                    pos.1 as f32 * super::consts::IMMEDIATE_NEIGHBOR_DIST + ground_pos_offset[1],
                ],
                ground_vel: starting_ground_vel,
                rest_mass: 1.0,
                object_index,
                id: MAX_PARTICLE_ID.fetch_add(1, Ordering::Relaxed),
                _a: 0,
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

pub struct SoftbodyState {
    // read back after each rk4 + culling
    particles: Vec<Particle>,
    objects: Vec<Object>, // fixed at 8192, since it's a uniform buffer

    // rk4 buffers & descsets
    particle_staging: Subbuffer<[Particle]>,
    pub particle_buf: Subbuffer<[Particle]>, // also used by the collision grid descset
    particle_buf_intermediate1: Subbuffer<[Particle]>,
    particle_buf_intermediate2: Subbuffer<[Particle]>,
    #[allow(unused)]
    forcesum: Subbuffer<[f32]>,

    euler_ds: Arc<PersistentDescriptorSet>,
    rk4_0_ds: Arc<PersistentDescriptorSet>,
    rk4_1_3_ds: Arc<PersistentDescriptorSet>,
    rk4_2_ds: Arc<PersistentDescriptorSet>,
    rk4_4_ds: Arc<PersistentDescriptorSet>,

    object_staging: Subbuffer<[Object]>,
    object_buf: Subbuffer<[Object]>,

    object_ds: Arc<PersistentDescriptorSet>,

    // collision buffers & descsets
    #[allow(unused)]
    pub spatial_lookup: Subbuffer<[[u32; 2]]>,
    #[allow(unused)]
    pub start_indices: Subbuffer<[u32]>,
    collision_update_ds: Arc<PersistentDescriptorSet>, // also used by the rk4 pipeline
}

impl SoftbodyState {
    // (how much should be allocated)
    const MAX_OBJECTS: u64 = 1024;
    pub const MAX_PARTICLES: u64 = 1 << 20;
    pub fn create(base: &BaseGpuState, pipelines: &SoftbodyComputePipelines) -> Self {
        let particle_staging = Buffer::new_slice::<Particle>(
            base.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            Self::MAX_OBJECTS,
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
        let rk4_0_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.rk4_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, particle_buf.clone()),
                WriteDescriptorSet::buffer(1, particle_buf.clone()),
                WriteDescriptorSet::buffer(2, particle_buf_intermediate1.clone()),
                WriteDescriptorSet::buffer(3, forcesum.clone()),
            ],
            [],
        )
        .unwrap();
        let rk4_1_3_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.rk4_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, particle_buf.clone()),
                WriteDescriptorSet::buffer(1, particle_buf_intermediate1.clone()),
                WriteDescriptorSet::buffer(2, particle_buf_intermediate2.clone()),
                WriteDescriptorSet::buffer(3, forcesum.clone()),
            ],
            [],
        )
        .unwrap();
        let rk4_2_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.rk4_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, particle_buf.clone()),
                WriteDescriptorSet::buffer(1, particle_buf_intermediate2.clone()),
                WriteDescriptorSet::buffer(2, particle_buf_intermediate1.clone()),
                WriteDescriptorSet::buffer(3, forcesum.clone()),
            ],
            [],
        )
        .unwrap();
        let rk4_4_ds = PersistentDescriptorSet::new(
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
        let spatial_lookup = Buffer::new_slice::<[u32; 2]>(
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
        let start_indices = Buffer::new_slice::<u32>(
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
        let collision_update_ds = PersistentDescriptorSet::new(
            &base.descriptor_set_allocator,
            pipelines.collision_grid_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, particle_buf.clone()),
                WriteDescriptorSet::buffer(1, spatial_lookup.clone()),
                WriteDescriptorSet::buffer(2, start_indices.clone()),
            ],
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
            rk4_0_ds,
            rk4_1_3_ds,
            rk4_2_ds,
            rk4_4_ds,
            object_staging,
            object_buf,
            object_ds,
            spatial_lookup,
            start_indices,
            collision_update_ds,
        }
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
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
                    size: (self.particles.len() * size_of::<Particle>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.particle_staging.as_bytes().clone(),
                    self.particle_buf_intermediate1.as_bytes().clone(),
                )
            })
            .unwrap()
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from_buf([BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (self.particles.len() * size_of::<Particle>()) as u64,
                    ..Default::default()
                }]),
                ..CopyBufferInfo::buffers(
                    self.particle_staging.as_bytes().clone(),
                    self.particle_buf_intermediate2.as_bytes().clone(),
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

    // should only be used at initialization
    // (or some other event that requires premature refresh of the collision grid)
    pub fn submit_initialize_cgrid(
        &self,
        base: &BaseGpuState,
        pipelines: &SoftbodyComputePipelines,
    ) -> Box<dyn GpuFuture> {
        let mut cmd_buf = base.create_primary_command_buffer();
        self.dispatch_update_compute_grid(pipelines, &mut cmd_buf);
        vulkano::sync::now(base.device.clone())
            .then_execute(base.queue.clone(), cmd_buf.build().unwrap())
            .unwrap()
            .boxed()
    }

    // does rk4 then update compute grid in one cbuf
    pub fn submit_per_frame_compute(
        &self,
        base: &BaseGpuState,
        pipelines: &SoftbodyComputePipelines,
    ) -> Box<dyn GpuFuture> {
        let mut cmd_buf = base.create_primary_command_buffer();
        unsafe {
            cmd_buf
                .write_timestamp(
                    base.query_pool.clone(),
                    crate::querybank::TOP_OF_PHYSICS,
                    PipelineStage::TopOfPipe,
                )
                .unwrap();
        }
        self.dispatch_rk4(pipelines, &mut cmd_buf);
        unsafe {
            cmd_buf
                .write_timestamp(
                    base.query_pool.clone(),
                    crate::querybank::RK4_AFTER,
                    PipelineStage::AllCommands,
                )
                .unwrap();
        }
        self.dispatch_update_compute_grid(pipelines, &mut cmd_buf);
        unsafe {
            cmd_buf
                .write_timestamp(
                    base.query_pool.clone(),
                    crate::querybank::GRID_UPDATE_AFTER,
                    PipelineStage::AllCommands,
                )
                .unwrap();
        }
        vulkano::sync::now(base.device.clone())
            .then_execute(base.queue.clone(), cmd_buf.build().unwrap())
            .unwrap()
            .boxed()
    }

    // unstable, strictly worse than rk4
    // well it probably is faster but not enough to justify the explosions
    // you don't want to use it is what i'm saying
    #[allow(unused)]
    #[allow(dependency_on_unit_never_type_fallback)]
    fn dispatch_euler(
        &self,
        pipelines: &SoftbodyComputePipelines,
        cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        cmd_buf
            .bind_pipeline_compute(pipelines.euler.clone())
            .unwrap()
            .push_constants(pipelines.rk4_pipeline_layout.clone(), 0, todo!())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                vec![
                    self.euler_ds.clone(),
                    self.object_ds.clone(),
                    self.collision_update_ds.clone(),
                ],
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap();
    }

    fn dispatch_rk4(
        &self,
        pipelines: &SoftbodyComputePipelines,
        cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        cmd_buf
            .bind_pipeline_compute(pipelines.rk4_0.clone())
            .unwrap()
            .push_constants(
                pipelines.rk4_pipeline_layout.clone(),
                0,
                Rk4PushConstants {
                    num_particles: self.particles.len() as u32,
                    ..super::consts::RK4_PUSH_CONSTS
                },
            )
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                vec![
                    self.rk4_0_ds.clone(),
                    self.object_ds.clone(),
                    self.collision_update_ds.clone(),
                ],
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipelines.rk4_1.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                self.rk4_1_3_ds.clone(),
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipelines.rk4_2.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                self.rk4_2_ds.clone(),
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipelines.rk4_3.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                self.rk4_1_3_ds.clone(),
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipelines.rk4_4.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.rk4_pipeline_layout.clone(),
                0,
                self.rk4_4_ds.clone(),
            )
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap();
    }

    // https://www.youtube.com/watch?v=rSKMYc1CQHE
    // sebastian lague <3
    // bitonic merge sort is clever i need to study more parallel algorithms
    fn dispatch_update_compute_grid(
        &self,
        pipelines: &SoftbodyComputePipelines,
        cmd_buf: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let pcs = CollisionGridPushConstants {
            num_particles: self.particles.len() as u32,
            grid_resolution: super::consts::GRID_RESOLUTION,
            group_width: 0,
            group_height: 0,
            step_index: 0,
        };
        cmd_buf
            .bind_pipeline_compute(pipelines.fill_lookup.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.collision_pipeline_layout.clone(),
                0,
                self.collision_update_ds.clone(),
            )
            .unwrap()
            .push_constants(pipelines.collision_pipeline_layout.clone(), 0, pcs.clone())
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap();
        cmd_buf
            .bind_pipeline_compute(pipelines.sort_lookup.clone())
            .unwrap();
        let num_pairs = self.particles.len().next_power_of_two() as u32 / 2;
        let num_stages = (num_pairs * 2).ilog2();
        for stage_index in 0..num_stages {
            for step_index in 0..=stage_index {
                let group_width = 1 << (stage_index - step_index);
                let group_height = 2 * group_width - 1;
                cmd_buf
                    .push_constants(
                        pipelines.collision_pipeline_layout.clone(),
                        0,
                        CollisionGridPushConstants {
                            group_width,
                            group_height,
                            step_index,
                            ..pcs
                        },
                    )
                    .unwrap()
                    .dispatch([num_pairs / 256, 1, 1])
                    .unwrap();
            }
        }
        cmd_buf
            .bind_pipeline_compute(pipelines.update_start_indices_1.clone())
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipelines.update_start_indices_2.clone())
            .unwrap()
            .dispatch([(self.particles.len() as u32).div_ceil(256), 1, 1])
            .unwrap();
    }

    // note that this function does NOT push the particles
    pub fn add_particles(&mut self, other: &mut Vec<Particle>, object: Object) {
        log::debug!(
            "Adding a softbody object with {} particles and metadata {:?}",
            other.len(),
            object
        );
        self.particles.append(other);
        self.objects.push(object);
    }
}

pub struct SoftbodyComputePipelines {
    // both of these are actually for rk4
    rk4_set_layout: Arc<DescriptorSetLayout>,
    objects_set_layout: Arc<DescriptorSetLayout>,

    collision_grid_set_layout: Arc<DescriptorSetLayout>,

    // rk4
    rk4_pipeline_layout: Arc<PipelineLayout>,
    euler: Arc<ComputePipeline>,
    rk4_0: Arc<ComputePipeline>,
    rk4_1: Arc<ComputePipeline>,
    rk4_2: Arc<ComputePipeline>,
    rk4_3: Arc<ComputePipeline>,
    rk4_4: Arc<ComputePipeline>,

    // collision
    collision_pipeline_layout: Arc<PipelineLayout>,
    fill_lookup: Arc<ComputePipeline>,
    sort_lookup: Arc<ComputePipeline>,
    update_start_indices_1: Arc<ComputePipeline>,
    update_start_indices_2: Arc<ComputePipeline>,
}

pub fn create_softbody_compute_pipelines(base: &mut BaseGpuState) -> SoftbodyComputePipelines {
    base.register_cache("rk4_0");
    base.register_cache("rk4_1");
    base.register_cache("rk4_2");
    base.register_cache("rk4_3");
    base.register_cache("rk4_4");
    base.register_cache("cgrid_fill_lookup");
    base.register_cache("cgrid_sort_lookup");
    base.register_cache("cgrid_update_start_indices_1");
    base.register_cache("cgrid_update_start_indices_2");
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
    let collision_grid_set_layout = DescriptorSetLayout::new(
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
    let rk4_pipeline_layout = PipelineLayout::new(
        base.device.clone(),
        PipelineLayoutCreateInfo {
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: size_of::<Rk4PushConstants>() as u32,
            }],
            set_layouts: vec![
                rk4_set_layout.clone(),
                objects_set_layout.clone(),
                collision_grid_set_layout.clone(),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!("spv/softbodyrk4_EULER.spv"))
                    .unwrap()
                    .borrow(),
            ),
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
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/softbodyrk4_RK4STAGE_0.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_0 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("rk4_0")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/softbodyrk4_RK4STAGE_1.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_1 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("rk4_1")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/softbodyrk4_RK4STAGE_2.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_2 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("rk4_2")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/softbodyrk4_RK4STAGE_3.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_3 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("rk4_3")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/softbodyrk4_RK4STAGE_4.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let rk4_4 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("rk4_4")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                rk4_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let collision_pipeline_layout = PipelineLayout::new(
        base.device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![collision_grid_set_layout.clone()],
            push_constant_ranges: vec![PushConstantRange {
                offset: 0,
                size: size_of::<CollisionGridPushConstants>() as u32,
                stages: ShaderStages::COMPUTE,
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/collision_grid_update_FILL_LOOKUP.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let fill_lookup = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("cgrid_fill_lookup")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                collision_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();

    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/collision_grid_update_SORT_LOOKUP.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let sort_lookup = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("cgrid_sort_lookup")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                collision_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();

    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/collision_grid_update_UPDATE_START_INDICES_1.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let update_start_indices_1 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("cgrid_update_start_indices_1")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                collision_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();
    let shader = unsafe {
        ShaderModule::new(
            base.device.clone(),
            ShaderModuleCreateInfo::new(
                vulkano::shader::spirv::bytes_to_words(include_bytes!(
                    "spv/collision_grid_update_UPDATE_START_INDICES_2.spv"
                ))
                .unwrap()
                .borrow(),
            ),
        )
        .unwrap()
    }
    .entry_point("main")
    .unwrap();
    let update_start_indices_2 = ComputePipeline::new(
        base.device.clone(),
        Some(base.get_cache("cgrid_update_start_indices_2")),
        ComputePipelineCreateInfo {
            stage: PipelineShaderStageCreateInfo::new(shader.clone()),
            ..ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader),
                collision_pipeline_layout.clone(),
            )
        },
    )
    .unwrap();

    SoftbodyComputePipelines {
        rk4_set_layout,
        objects_set_layout,
        collision_grid_set_layout,
        rk4_pipeline_layout,
        euler,
        rk4_0,
        rk4_1,
        rk4_2,
        rk4_3,
        rk4_4,
        collision_pipeline_layout,
        fill_lookup,
        sort_lookup,
        update_start_indices_1,
        update_start_indices_2,
    }
}
