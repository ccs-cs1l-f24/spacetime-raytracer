use winit::event::{ElementState, KeyEvent};

#[derive(Default)]
pub struct Keyboard {
    pub up: bool,
    pub left: bool,
    pub down: bool,
    pub right: bool,
    pub z: bool,
    pub x: bool,
}

impl Keyboard {
    pub fn process_key_event(
        &mut self,
        KeyEvent {
            state,
            text,
            // repeat,
            ..
        }: KeyEvent,
    ) {
        if let Some(text) = text {
            let state = match state {
                ElementState::Pressed => true,
                ElementState::Released => false,
            };
            match text.as_str() {
                "w" => self.up = state,
                "a" => self.left = state,
                "s" => self.down = state,
                "d" => self.right = state,
                "z" => self.z = state,
                "x" => self.x = state,
                _ => {}
            }
        }
    }
}
