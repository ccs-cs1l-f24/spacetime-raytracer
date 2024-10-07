use log::{Level, Log};

// the most stripped down logger
// that it's possible to have
struct Logger;

impl Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Debug
    }

    fn log(&self, record: &log::Record) {
        println!("[{}] {}", record.level(), record.args())
    }

    fn flush(&self) {}
}

static LOGGER: Logger = Logger;

pub fn initialize() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
}
