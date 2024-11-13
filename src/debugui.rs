use std::{collections::VecDeque, sync::Arc, time::Duration};

use vulkano::{image::view::ImageView, sync::GpuFuture};

use crate::{boilerplate::BaseGpuState, querybank::FramePerfStats};

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
        Self { max_fps: 72 }
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
    frame_times: VecDeque<Duration>,
    sorted_frame_times: Vec<Duration>,

    // internal state
    max_fps_input: String,
}

impl DebugUiState {
    pub fn add_frame_time(&mut self, time: Duration) {
        self.frame_times.push_back(time);
        if self.frame_times.len() > 2000 {
            self.frame_times.pop_front();
        }
        self.sorted_frame_times = self.frame_times.iter().cloned().collect();
        self.sorted_frame_times.sort();
    }

    // call before each render attempt
    pub fn start_gui(&mut self, perf: FramePerfStats) {
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
                        "Frame Duration Minimum: {:?}",
                        self.config.fps_duration()
                    ));
                    ui.label(format!(
                        // there is guaranteed to be at least one element in self.frame_times and self.sorted_frame_times
                        "Last Frame Time: {:?}\nAverage: {:?}\n1% low: {:?}\n0.1% low: {:?}",
                        self.frame_times.back().unwrap(),
                        self.frame_times
                            .range(self.sorted_frame_times.len().checked_sub(50).unwrap_or(0)..)
                            .sum::<Duration>()
                            / 50.min(self.sorted_frame_times.len() as u32),
                        self.sorted_frame_times[self.sorted_frame_times.len()
                            - self.sorted_frame_times.len() / 100
                            - 1],
                        self.sorted_frame_times.last().unwrap(),
                    ));
                    ui.label(format!("{}", perf));
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
    let mut frame_times = VecDeque::new();
    frame_times.push_back(Duration::new(0, 0));
    DebugUiState {
        gui,
        config: Default::default(),
        frame_times,
        sorted_frame_times: vec![Duration::new(0, 0)],
        max_fps_input: Default::default(),
    }
}
