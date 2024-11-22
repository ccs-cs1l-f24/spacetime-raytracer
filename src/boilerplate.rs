use std::{
    collections::HashMap, fs::OpenOptions, io::Write, path::PathBuf, str::FromStr, sync::Arc,
};

use log::Level;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::{
        StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features, Queue, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageUsage},
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        cache::{PipelineCache, PipelineCacheCreateInfo},
        graphics::viewport::Viewport,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryType},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        FullScreenExclusive, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::GpuFuture,
    VulkanError, VulkanLibrary,
};

pub const CACHE_DIR: &str = "cache";

// ALL of the render passes my application uses are here
// mostly so i can centralize framebuffer regeneration
pub struct RpassManager {
    pub main_pass: RpassWrapper,
    // ... other passes
    // (egui_winit_vulkano passes are managed separately)
}

impl RpassManager {
    fn create(device: Arc<Device>, swapchain_manager: &SwapchainManager) -> Self {
        Self {
            main_pass: {
                let rpass = vulkano::single_pass_renderpass!(
                    device.clone(),
                    attachments: {
                        color: {
                            format: swapchain_manager.swapchain_format(),
                            samples: 1,
                            load_op: Clear,
                            store_op: Store,
                        }
                    },
                    pass: {
                        color: [color],
                        depth_stencil: {},
                    }
                )
                .unwrap();
                RpassWrapper::create(swapchain_manager, rpass)
            },
        }
    }

    fn update_framebuffers(&mut self, swapchain_manager: &SwapchainManager) {
        RpassWrapper::update_framebuffers(&mut self.main_pass, swapchain_manager);
    }
}

// we have (frames in flight) framebuffers per render pass
// that all need to be updated
pub struct RpassWrapper {
    pub rpass: Arc<RenderPass>,
    // ok technically this is just the first subpass
    // if future me is using a multi-stage render pass
    // well future me would probably want to restructure this in that case
    // but it isn't incompatible with a >1 subpass approach to store the first one here
    pub subpass: Subpass,
    pub framebuffers: Vec<Arc<Framebuffer>>,
}

impl RpassWrapper {
    pub fn create(swapchain_manager: &SwapchainManager, rpass: Arc<RenderPass>) -> Self {
        let subpass = rpass.clone().first_subpass();
        let mut s = Self {
            rpass,
            subpass,
            framebuffers: vec![],
        };
        RpassWrapper::update_framebuffers(&mut s, swapchain_manager);
        s
    }
    pub fn update_framebuffers(s: &mut Self, swapchain_manager: &SwapchainManager) {
        s.framebuffers = swapchain_manager.create_framebuffers_for(s.rpass.clone());
    }
}

// big thanks to
// https://github.com/vulkano-rs/vulkano/blob/master/examples/triangle-v1_3/main.rs
// for having most of the boilerplate in one concise-ish place

// bound to a window
// has all the base vulkan state
// pipelines and shaders are stored elsewhere
pub struct BaseGpuState {
    #[allow(unused)]
    pub instance: Arc<Instance>,
    #[allow(unused)]
    pub surface: Arc<Surface>,
    #[allow(unused)]
    pub debug: DebugUtilsMessenger, // gotta make sure it doesn't die early
    #[allow(unused)]
    pub physical_device: Arc<PhysicalDevice>,
    pub queue_family_index: u32,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,

    pub query_pool: Arc<QueryPool>,
    pub query_results: [u64; crate::querybank::NUM_QUERIES as usize],

    pub swapchain_manager: SwapchainManager,

    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: StandardDescriptorSetAllocator,

    pub pipeline_caches: HashMap<&'static str, Arc<PipelineCache>>,

    pub rpass_manager: RpassManager,
}

impl BaseGpuState {
    // convenience method
    pub fn create_primary_command_buffer(
        &self,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    // pipeline cache stuff

    pub fn get_cache(&self, name: &'static str) -> Arc<PipelineCache> {
        self.pipeline_caches.get(name).unwrap().clone()
    }

    pub fn register_cache(&mut self, name: &'static str) {
        let path = PathBuf::from_str(CACHE_DIR)
            .unwrap()
            .join(name)
            .with_extension("bin");
        let initial_data = std::fs::read(path).unwrap_or(vec![]);
        let cache = unsafe {
            PipelineCache::new(
                self.device.clone(),
                PipelineCacheCreateInfo {
                    initial_data,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        assert!(
            self.pipeline_caches.insert(name, cache).is_none(),
            "You already registered that pipeline cache!!"
        );
    }

    pub fn write_caches_out(&self) {
        for name in self.pipeline_caches.keys() {
            self.write_cache_out(name);
        }
    }

    fn write_cache_out(&self, name: &str) {
        let path = PathBuf::from_str(CACHE_DIR)
            .unwrap()
            .join(name)
            .with_extension("bin");
        let data = self.pipeline_caches.get(name).unwrap().get_data().unwrap();
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(path)
            .unwrap()
            .write_all(&data)
            .unwrap();
    }

    // profiling query pool / timestamp stuff

    pub fn update_query_results(&mut self) {
        let mut next = [0u64; crate::querybank::NUM_QUERIES as usize * 2];
        self.query_pool
            .get_results(
                0..crate::querybank::NUM_QUERIES,
                &mut next,
                vulkano::query::QueryResultFlags::WITH_AVAILABILITY,
            )
            .unwrap();
        for i in 0..self.query_results.len() {
            if next[i * 2 + 1] != 0 {
                self.query_results[i] = next[i * 2];
            }
        }
    }

    pub fn reset_query_pool(&self) {
        let mut cbuf = self.create_primary_command_buffer();
        unsafe {
            cbuf.reset_query_pool(self.query_pool.clone(), 0..crate::querybank::NUM_QUERIES)
                .unwrap();
        }
        let cbuf = cbuf.build().unwrap();
        vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cbuf)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    // handles both swapchain and framebuffer resizing
    pub fn resize(&mut self, image_extent: [u32; 2]) {
        self.swapchain_manager.recreate(image_extent);
        self.rpass_manager
            .update_framebuffers(&self.swapchain_manager);
    }
}

pub struct SwapchainManager {
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    image_views: Vec<Arc<ImageView>>,
    frames_in_flight_futures: Vec<Option<Box<dyn GpuFuture>>>,
    pub viewport: Viewport,
}

impl SwapchainManager {
    pub fn create(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window_size: [u32; 2],
        desired_frames_in_flight: u32,
    ) -> Self {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(
                &surface,
                SurfaceInfo {
                    // no fullscreen exclusive!! i don't like fullscreen exclusive it's sad
                    // unfortunately i need to add an extension to explicitly deny it
                    // so instead i'm having winit ban it from happening
                    // in the meantime we can have a sad default and overlong comment
                    full_screen_exclusive: FullScreenExclusive::Default,
                    // i don't want to bother thinking about present modes
                    // FIFO is like fine right
                    present_mode: None,
                    ..Default::default()
                },
            )
            .unwrap();

        let formats = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap();
        // just take the first one
        // technically we should want bgraunorm (non-srgb) cause of egui
        // but i don't think we care that much
        let (image_format, color_space) = formats[0];
        log::info!(
            "Choosing image format {:?}, {:?} for the swapchain out of {:?}",
            image_format,
            color_space,
            formats
        );
        let frames_in_flight = surface_capabilities
            .min_image_count
            .max(desired_frames_in_flight);
        log::info!("We'll be using {:?} frames in flight", frames_in_flight);
        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: frames_in_flight,
                image_format,
                image_extent: window_size,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                // idk how to make an informed choice about the composite alpha
                // so i'll use the first one supported
                // pro strats here
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                present_mode: PresentMode::Fifo, // ol' reliable
                ..Default::default()
            },
        )
        .unwrap();
        let frames_in_flight_futures = (0..frames_in_flight).map(|_| None).collect();
        let mut manager = Self {
            swapchain,
            images,
            image_views: vec![],
            frames_in_flight_futures,
            viewport: Viewport::default(),
        };
        SwapchainManager::set_image_views_and_viewport(&mut manager);
        manager
    }

    // pub fn frames_count(&self) -> usize {
    //     self.image_views.len()
    // }

    pub fn set_frame_in_flight_future(&mut self, future: Box<dyn GpuFuture>, index: u32) {
        self.frames_in_flight_futures[index as usize] = Some(future);
    }

    // if there's no frame in flight future, just say the frame (not in flight cause it didn't actually happen lmao) already happened
    pub fn take_frame_in_flight_future(&mut self, index: u32) -> Box<dyn GpuFuture> {
        self.frames_in_flight_futures[index as usize]
            .take()
            .unwrap_or(vulkano::sync::now(self.swapchain.device().clone()).boxed())
    }

    pub fn present_info(&self, image_index: u32) -> SwapchainPresentInfo {
        SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index)
    }

    // outsourced to by the RpassWrappers
    fn create_framebuffers_for(&self, rpass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
        self.image_views
            .iter()
            .cloned()
            .map(|view| {
                Framebuffer::new(
                    rpass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    pub fn get_image_view(&self, index: u32) -> Arc<ImageView> {
        self.image_views[index as usize].clone()
    }

    pub fn swapchain_format(&self) -> Format {
        self.swapchain.image_format()
    }

    // the in-flight images should be like fine i think
    fn recreate(&mut self, image_extent: [u32; 2]) {
        log::debug!("Resizing swapchain to {:?}", image_extent);
        let (swapchain, images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent,
                ..self.swapchain.create_info()
            })
            .unwrap();
        self.swapchain = swapchain;
        self.images = images;
        SwapchainManager::set_image_views_and_viewport(self);
    }

    // gracefully handles if the swapchain is out of date
    // is ungraceful if the error is caused by anything else wow bad manners smh
    // passes on the suboptimal flag with a logged warning if it's true
    pub fn acquire_image(&self) -> Result<(u32, bool, SwapchainAcquireFuture), ()> {
        let (image_index, suboptimal, acquire_future) =
            vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None).map_err(
                |err| {
                    let err = err.unwrap();
                    match err {
                        VulkanError::OutOfDate => log::error!("Swapchain out of date!"),
                        err => panic!("Swapchain image cannot be acquired; {:?}", err),
                    }
                },
            )?;
        if suboptimal {
            log::warn!("The swapchain image is suboptimal!!");
        }
        Ok((image_index, suboptimal, acquire_future))
    }

    fn set_image_views_and_viewport(manager: &mut Self) {
        let extent = manager.images[0].extent();
        manager.viewport.extent[0] = extent[0] as f32;
        manager.viewport.extent[1] = extent[1] as f32;
        manager.image_views = manager
            .images
            .iter()
            .cloned() // it's fine we're just cloning arcs
            .map(|image| ImageView::new_default(image).unwrap())
            .collect();
    }
}

pub fn create_gpu_state(window: Arc<winit::window::Window>) -> BaseGpuState {
    let library = VulkanLibrary::new().unwrap();

    let mut instance_extensions = Surface::required_extensions(&window);
    instance_extensions.ext_debug_utils = true;

    // enumerate possible layers
    // for fun :)
    log::info!("List of Vulkan layers available to use:");
    let layers = library.layer_properties().unwrap();
    let layers = layers
        .filter(|l| {
            log::info!("- {} : {}", l.name(), l.description());
            l.name() == "VK_LAYER_KHRONOS_validation"
        })
        .map(|l| l.name().to_owned())
        .collect();
    log::info!("Vulkan layers to be used: {:?}", layers);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, // for that macos support
            enabled_extensions: instance_extensions,
            enabled_layers: layers,
            ..Default::default()
        },
    )
    .unwrap();

    // right so none of this is strictly relevant to the code below but i must tell this saga somewhere
    // i'm very mad at vulkano as i write this
    // in short, they were constructing queue/cbuf label slices out of null pointers
    // this pull request fixes the problem https://github.com/vulkano-rs/vulkano/pull/2490
    // but 0.34.1, the as-i-write-this latest version, doesn't have this PR merged
    // despite that the PR was made in fucking march and it's like july
    // so i forked vulkano 0.34.1 and added the change myself
    // sorry vulkano devs i love you but this bowled me over
    // literally on the floor rn :P
    let debug = DebugUtilsMessenger::new(
        instance.clone(),
        DebugUtilsMessengerCreateInfo {
            message_severity: DebugUtilsMessageSeverity::ERROR
                | DebugUtilsMessageSeverity::WARNING
                | DebugUtilsMessageSeverity::INFO
                | DebugUtilsMessageSeverity::VERBOSE,
            message_type: DebugUtilsMessageType::VALIDATION
                | DebugUtilsMessageType::GENERAL
                | DebugUtilsMessageType::PERFORMANCE,
            ..DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                // don't make any calls to Vulkan here
                // or spooky UNSAFE things will happen
                // and that's bad
                DebugUtilsMessengerCallback::new(|severity, kind, data| {
                    let DebugUtilsMessengerCallbackData {
                        message_id_name,
                        message_id_number: _,
                        message,
                        queue_labels,
                        cmd_buf_labels,
                        objects,
                        ..
                    } = data;
                    let queue_labels = queue_labels.map(|info| info.label_name).collect::<Vec<_>>();
                    let cmd_buf_labels = cmd_buf_labels
                        .map(|info| info.label_name)
                        .collect::<Vec<_>>();
                    let objects = objects
                        .map(|info| (info.object_name, info.object_type))
                        .collect::<Vec<_>>();

                    let severity = if severity.intersects(DebugUtilsMessageSeverity::ERROR) {
                        Level::Error
                    } else if severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                        Level::Warn
                    } else if severity.intersects(DebugUtilsMessageSeverity::INFO) {
                        Level::Info
                    } else if severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                        Level::Debug
                    } else {
                        panic!("Invalid severity kind {:?}", severity);
                    };
                    let kind = if kind.intersects(DebugUtilsMessageType::VALIDATION) {
                        "VALID"
                    } else if kind.intersects(DebugUtilsMessageType::GENERAL) {
                        "GEN"
                    } else if kind.intersects(DebugUtilsMessageType::PERFORMANCE) {
                        "PERF"
                    } else {
                        panic!("Invalid message type {:?}", kind)
                    };

                    log::log!(
                        severity,
                        "[{}|{:?}|{:?}|{:?}] {}: {}",
                        kind,
                        queue_labels,
                        cmd_buf_labels,
                        objects,
                        message_id_name.unwrap_or("unknown"),
                        message,
                    );
                })
            })
        },
    )
    .unwrap();

    // make the surface
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    // device selection
    let desired_device_extensions = DeviceExtensions {
        khr_acceleration_structure: true,
        khr_ray_query: true,
        // khr_ray_tracing_pipeline: true,
        ..DeviceExtensions::empty()
    };
    let necessary_device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_host_query_reset: true,
        ..DeviceExtensions::empty()
    };
    let physical_devices = instance
        .enumerate_physical_devices()
        .unwrap()
        .collect::<Vec<_>>();
    log::debug!(
        "Considering physical devices: {:?}",
        physical_devices
            .iter()
            .map(|d| &d.properties().device_name)
            .collect::<Vec<_>>()
    );
    let necessary_filter = |p_d: Arc<PhysicalDevice>| {
        // basic bitch approach to queue families
        // select one family for all things
        // the family with the graphics and compute and all that
        // which has to exist
        // if the device doesn't support presenting to the surface then discard it
        // you're in charge you're the boss you can do whatever you want
        let family = p_d
            .queue_family_properties()
            .iter()
            .enumerate()
            .find(|(index, props)| {
                props
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER)
                    && p_d.surface_support(*index as u32, &surface).unwrap()
                    && p_d.properties().timestamp_compute_and_graphics
            })?;
        if !p_d
            .supported_extensions()
            .contains(&necessary_device_extensions)
        {
            return None;
        }
        Some((p_d.clone(), family.0 as u32))
    };
    let (physical_device, queue_family_index) = physical_devices
        .iter()
        .cloned() // it's fineee we're just cloning arcs
        .filter_map(necessary_filter)
        .max_by_key(|(p_d, _)| {
            // yay taking into account client preferences and all that jazz ok cool
            let props = p_d.properties();
            if let Some(name) = &Option::<String>::None {
                // cfg.base.name
                let p_d_name = &p_d.properties().device_name;
                log::info!(
                    "Physical device named {:?} is desired; examining device named {:?}",
                    name,
                    p_d_name
                );
                (p_d_name == name) as u8
            } else {
                // how to weigh the device properties against each other
                let (rtx, discrete) = (4, 3);
                p_d.supported_extensions()
                    .contains(&desired_device_extensions) as u8
                    * rtx
                    + match props.device_type {
                        PhysicalDeviceType::DiscreteGpu => discrete,
                        PhysicalDeviceType::IntegratedGpu => 2,
                        PhysicalDeviceType::VirtualGpu => 1,
                        _ => 0,
                    }
            }
        })
        .expect("Could not find a single physical device supporting that surface!");

    let device_extensions = necessary_device_extensions
        | if physical_device
            .supported_extensions()
            .contains(&desired_device_extensions)
        {
            desired_device_extensions
        } else {
            DeviceExtensions::empty()
        };
    let device_features = Features {
        acceleration_structure: device_extensions.khr_acceleration_structure,
        ray_query: device_extensions.khr_ray_query,
        fill_mode_non_solid: true,
        host_query_reset: true,
        ..Default::default()
    };
    // if !device_extensions.khr_acceleration_structure {
    //     panic!("aaaahhhhhhhhh we need raytracing for this to workk")
    // }

    log::info!(
        "Chose the device: {} ({:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            enabled_features: device_features,
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap(); // just one queue

    let query_pool = QueryPool::new(
        device.clone(),
        QueryPoolCreateInfo {
            query_count: crate::querybank::NUM_QUERIES,
            ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        },
    )
    .unwrap();

    let swapchain_manager = SwapchainManager::create(
        device.clone(),
        surface.clone(),
        window.inner_size().into(),
        2, //cfg.base.desired_frames_in_flight,
    );

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo {
            set_count: 32,
            update_after_bind: false,
            ..Default::default()
        },
    );

    let rpass_manager = RpassManager::create(device.clone(), &swapchain_manager);

    BaseGpuState {
        instance,
        debug,
        surface,
        physical_device,
        queue_family_index,
        device,
        queue,
        query_pool,
        query_results: [0; crate::querybank::NUM_QUERIES as usize],
        swapchain_manager,
        memory_allocator,
        command_buffer_allocator,
        descriptor_set_allocator,
        pipeline_caches: HashMap::new(),
        rpass_manager,
    }
}
