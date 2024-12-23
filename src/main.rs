use std::{sync::Arc, time::Instant};

mod boilerplate;
mod debugui;
mod keyboard;
mod logimpl;
mod querybank;

mod twoplusone;

fn main() {
    logimpl::initialize();
    log::info!("Starting!!");

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    // WHY does winit insist on changing their API every goddamn update
    // just kidding i love you winit devs but still it gets confusing
    // trying to keep up with the massive redesigns
    // even though to be fair the new designs are like always better
    let mut app = App {
        prev_frame_start: Instant::now(),
        prev_frame_stats: Default::default(),
        keyboard: keyboard::Keyboard::default(),
        init_state: None,
    };
    event_loop.run_app(&mut app).unwrap();

    log::info!("Closing time :)");
}

struct InitState {
    main_window: Arc<winit::window::Window>,
    base_gpu: boilerplate::BaseGpuState,
    debug_ui_state: debugui::DebugUiState,

    // pipelines
    pipeline_manager: twoplusone::PipelineManager,

    // simulation/game state + buffers/descriptor sets
    world: twoplusone::World,

    // unlike render ops, we can only have one physics op in flight at once
    // since each one depends on the previous
    // so we take and block on it before submitting the next frame's render op
    in_flight_physics: Option<Box<dyn vulkano::sync::GpuFuture>>,

    // checked and reset at the start of each frame render
    // used so that "suboptimal" presents don't delay a frame unnecessarily with swapchain recreation
    recreate_swapchain: bool,
}

struct App {
    // for accurate profiling, since the "start" in resumetimereached often refers to something else
    prev_frame_start: Instant,
    prev_frame_stats: querybank::FramePerfStats,
    keyboard: keyboard::Keyboard,
    // i hate that this has to be option
    // but there's no way to initialize the vulkan context without the surface
    // which is to say the window... which the event loop has to be running to create
    init_state: Option<InitState>,
}

impl winit::application::ApplicationHandler for App {
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        match cause {
            winit::event::StartCause::ResumeTimeReached {
                start: _,
                requested_resume,
            } => {
                let state = self
                    .init_state
                    .as_mut()
                    .expect("Init state really should exist by now");
                event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                    // if the previous frame took longer than fps_duration to happen
                    // then take the L and set the target frame time to be now
                    (requested_resume + state.debug_ui_state.config.fps_duration())
                        .max(Instant::now()),
                ));
                state
                    .debug_ui_state
                    .add_frame_time(self.prev_frame_start.elapsed());
                state.world.update_camera(
                    &self.keyboard,
                    self.prev_frame_start.elapsed().as_secs_f32(),
                );
                self.prev_frame_start = Instant::now();
                state.main_window.request_redraw();
            }
            winit::event::StartCause::WaitCancelled {
                start: _,
                requested_resume,
            } => {
                // re-request the requested resume
                // so that we can have the frame logic happen
                // at when that resume time is reached
                event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(requested_resume.expect("There shouldn't not be a requested resume time... if there is uh check in please")));
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // ok so the winit docs say to only initialize the graphics context et al
        // after the app is first resumed
        // because android
        // and even though i don't care about supporting android this is a nice place to put it
        // since it's cleanly separated from the frame/redraw logics of new events and window events
        if let None = self.init_state {
            log::debug!("Initializing the window and graphics context!!");
            let main_window = Arc::new(
                event_loop
                    .create_window(
                        winit::window::WindowAttributes::default()
                            .with_resizable(true)
                            // no fullscreen by default cause i don't like it
                            .with_fullscreen(None)
                            .with_title("Special Relativity"),
                    )
                    .unwrap(),
            );
            let mut base_gpu = boilerplate::create_gpu_state(main_window.clone());
            base_gpu.reset_query_pool(); // for some reason it doesn't start flushed
            log::debug!("Now we have a graphics context and window secured :D"); // many smiles here
            let debug_ui_state = debugui::create_debug_ui_state(event_loop, &base_gpu);
            log::debug!("Debug UI created :D");
            let pipeline_manager = twoplusone::create_pipeline_manager(&mut base_gpu); // mutates the pipeline cache registry
            log::debug!("Pipelines created :D");
            let world = twoplusone::create_world(&base_gpu, &pipeline_manager);
            log::debug!("World created :D");
            // initializing things :D
            use vulkano::sync::GpuFuture;
            world
                .softbody_state // this way the first physics frame isn't working from an uninitialized cgrid (SLOW)
                .submit_initialize_cgrid(&base_gpu, &pipeline_manager.softbody_compute)
                // .join(
                //     // this way the edge map starts out empty
                //     // and we can have a use->clear->populate loop
                //     world
                //         .worldline_update_softbodies_state
                //         .submit_clear_edge_map(
                //             &base_gpu,
                //             &pipeline_manager.worldline_update_softbodies,
                //         ),
                // )
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(Instant::now()));
            let init_state = InitState {
                main_window,
                base_gpu,
                debug_ui_state,
                pipeline_manager,
                world,
                in_flight_physics: None,
                recreate_swapchain: false,
            };
            self.init_state = Some(init_state);

            // initialization doesn't count as the first frame!!
            // this way the 0.1% lows aren't thrown off quite so badly :)
            self.prev_frame_start = Instant::now();
        }
    }

    // conspicuously absent is an implementation of suspended
    // this is because it is relevant only for iOS, Android, and Web
    // none of which are platforms i'm aiming to support here
    // i don't want to add destroy-then-rebuild-the-graphics-context boilerplate
    // for platforms i'm not targeting

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(InitState {
            debug_ui_state,
            main_window,
            ..
        }) = &mut self.init_state
        {
            if debug_ui_state.on_window_event(&main_window, &event) {
                return;
            }
        }

        match event {
            winit::event::WindowEvent::Resized(size) => {
                // proactively recreate the swapchain and update the framebuffers
                log::debug!("Resizing window to {:?}", size);
                if let Some(InitState { base_gpu, .. }) = &mut self.init_state {
                    base_gpu.resize(size.into());
                } else {
                    log::warn!("Resize event called with init state uninitialized")
                }
            }
            winit::event::WindowEvent::CloseRequested => {
                // be graceful
                // exit the event loop & clean up graphics state
                // save progress/levels/make sure the persistent state is fine
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                if let Some(InitState {
                    main_window,
                    base_gpu,
                    debug_ui_state,
                    pipeline_manager,
                    world,
                    in_flight_physics,
                    recreate_swapchain,
                }) = &mut self.init_state
                {
                    if Into::<[u32; 2]>::into(main_window.inner_size()).contains(&0) {
                        log::debug!(
                            "I'm not drawing to a window with inner size {:?}",
                            main_window.inner_size()
                        );
                        return;
                    }
                    debug_ui_state.start_gui(self.prev_frame_stats.clone());

                    if *recreate_swapchain {
                        base_gpu.resize(main_window.inner_size().into());
                        *recreate_swapchain = false;
                    }

                    let (present_image_index, suboptimal, present_image_acquire_future) =
                        match base_gpu.swapchain_manager.acquire_image() {
                            Ok(stuff) => stuff,
                            Err(()) => {
                                base_gpu.resize(main_window.inner_size().into());
                                base_gpu
                                    .swapchain_manager
                                    .acquire_image()
                                    .expect("What??? I just recreated the swapchain come onnn")
                            }
                        };
                    if suboptimal {
                        // don't delay this frame by recreating the swapchain, do it later
                        *recreate_swapchain = true;
                    }

                    // wait on the in-flight physics before composing/submitting the next render op
                    if let Some(blocking_physics) = in_flight_physics.take() {
                        blocking_physics
                            .then_signal_fence_and_flush()
                            .unwrap()
                            .wait(None)
                            .unwrap();
                    }

                    base_gpu.update_query_results();
                    self.prev_frame_stats = crate::querybank::get_frame_perf_stats(&base_gpu);
                    base_gpu.reset_query_pool(); // ALL QUERIES SHOULD GO AFTER HERE

                    let worldline_update_future = world
                        .worldline_update_softbodies_state
                        .submit_update_worldlines(
                            &base_gpu,
                            &pipeline_manager.worldline_update_softbodies,
                            world.softbody_state.num_particles() as u32,
                        );

                    // scene rendering for this frame
                    // (everything except the debug ui)
                    let mut cmd_buf = base_gpu.create_primary_command_buffer();
                    twoplusone::softbody::point_render_nr::render(
                        present_image_index,
                        main_window.inner_size().width as f32
                            / main_window.inner_size().height as f32,
                        base_gpu,
                        &mut cmd_buf,
                        &world.softbody_state,
                        &pipeline_manager.point_pipelines,
                        world.cam_ground_pos,
                        world.zoom,
                    );
                    let cmd_buf = cmd_buf.build().unwrap();

                    let mut prev_frame_in_flight_future = base_gpu
                        .swapchain_manager
                        .take_frame_in_flight_future(present_image_index);
                    // make sure to clean up the fences so they don't accumulate
                    prev_frame_in_flight_future.cleanup_finished();

                    // import this otherwise i can't build the future :P
                    use vulkano::sync::GpuFuture;
                    // this future does the scene rendering
                    let future = prev_frame_in_flight_future
                        .join(present_image_acquire_future)
                        .join(worldline_update_future)
                        .then_execute(base_gpu.queue.clone(), cmd_buf)
                        .unwrap();
                    // this future does the debug ui, after everything else is rendered
                    let future = debug_ui_state.render_gui(
                        future.boxed(),
                        base_gpu
                            .swapchain_manager
                            .get_image_view(present_image_index),
                    );
                    // this future represents when this frame will have been presented
                    let future = future
                        .then_swapchain_present(
                            base_gpu.queue.clone(),
                            base_gpu.swapchain_manager.present_info(present_image_index),
                        )
                        .then_signal_fence_and_flush();
                    match future.map_err(vulkano::Validated::unwrap) {
                        Ok(future) => {
                            base_gpu
                                .swapchain_manager
                                .set_frame_in_flight_future(future.boxed(), present_image_index);
                        }
                        Err(vulkano::VulkanError::OutOfDate) => {
                            *recreate_swapchain = true;
                        }
                        Err(e) => {
                            panic!("Failed to flush the main render future; {:?}", e);
                        }
                    }
                    main_window.pre_present_notify(); // we've submitted a render op, let's notify them :)

                    // submit the physics for the next frame :)
                    if !self.keyboard.pause {
                        *in_flight_physics = Some(world.softbody_state.submit_per_frame_compute(
                            &base_gpu,
                            &pipeline_manager.softbody_compute,
                        ));
                    }
                } else {
                    unreachable!("Init state really must exist by now")
                }
            }
            // TODO: mouse input
            // winit::event::WindowEvent::CursorMoved { device_id, position } => {}
            // winit::event::WindowEvent::MouseInput { state, button, .. } => {}
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                self.keyboard.process_key_event(event);
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // destroy the vulkan context and assets before the window goes
        // can do other cleanup after the loop exits
        if let Some(InitState { base_gpu, .. }) = &self.init_state {
            base_gpu.write_caches_out();
        }
        log::debug!("Exiting the event loop!");
    }
}
