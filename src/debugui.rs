use std::{sync::Arc, time::Duration};

use vulkano::{image::view::ImageView, sync::GpuFuture};

use crate::boilerplate::BaseGpuState;

// the rendering and simulation options
// that should be tweakable live from the debug ui
pub struct HotswapConfig {
    pub max_fps: u32,
}

impl HotswapConfig {
    pub fn fps_duration(&self) -> Duration {
        Duration::from_micros((1000000.0 / self.max_fps as f32) as u64)
    }
}

impl Default for HotswapConfig {
    fn default() -> Self {
        Self {
            max_fps: 144,
        }
    }
}

// we'll just render every frame to the swapchain image, on top of the scene et al after the fact
// also holds important config data
pub struct DebugUiState {
    // contains the egui context, winit integration, and vulkano state necessary to render
    // all in one package!
    gui: egui_winit_vulkano::Gui,

    pub config: HotswapConfig,

    // data used to compose the gui
    // TODO average / 1% / 0.1%
    pub time_since_last_frame: std::time::Duration,

    // internal state
    max_fps_input: String,
}

impl DebugUiState {
    // call before each render attempt
    pub fn start_gui(&mut self) {
        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            egui::Window::new("Debug UI")
                // .default_size([300.0, 500.0])
                .max_size([300.0, 500.0])
                .default_pos([0.0, 0.0])
                .title_bar(true)
                .movable(true)
                .show(&ctx, |ui| {
                    ui.heading("Profiling");
                    ui.label(format!(
                        "Current FPS: {}",
                        (1000000.0 / self.time_since_last_frame.as_micros() as f32).ceil() as u64
                    ));
                    ui.separator();
                    ui.heading("Render Settings");
                    // ui.horizontal(|ui| {
                    //     ui.radio_value(&mut self.config.fullscreen, false, "Windowed");
                    //     ui.radio_value(&mut self.config.fullscreen, true, "Fullscreen");
                    // });
                    ui.horizontal(|ui| {
                        if ui.button("Max FPS").clicked() {
                            match self.max_fps_input.parse() {
                                Ok(fps) => self.config.max_fps = fps,
                                Err(e) => log::error!(
                                    "Invalid max fps value {:?}; {:?}",
                                    self.max_fps_input,
                                    e
                                ),
                            };
                        }
                        ui.text_edit_singleline(&mut self.max_fps_input);
                    });
                });
        })
    }

    // for some reason, egui_winit_vulkano's only api for drawing
    // gives a future instead of a command buffer
    // i want to be able to chain my futures together in one big statement
    // and now i can't >:(
    // or i guess i could rewrite that part of the library but i don't want to bother
    // since it's not strictly necessary
    pub fn render_gui(
        &mut self,
        wait_on: Box<dyn GpuFuture>,
        onto: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        self.gui.draw_on_image(wait_on, onto)
    }

    // returns true if egui claims exclusive control of this event
    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.gui.update(window, event)
    }
}

pub fn create_debug_ui_state(
    event_loop: &winit::event_loop::ActiveEventLoop,
    base: &BaseGpuState,
) -> DebugUiState {
    let gui = egui_winit_vulkano::Gui::new(
        event_loop,
        base.surface.clone(),
        base.queue.clone(),
        base.swapchain_manager.swapchain_format(),
        egui_winit_vulkano::GuiConfig {
            allow_srgb_render_target: true,
            is_overlay: true,
            samples: vulkano::image::SampleCount::Sample1,
        },
    );
    DebugUiState {
        gui,
        config: Default::default(),
        time_since_last_frame: Default::default(),
        max_fps_input: Default::default(),
    }
}
