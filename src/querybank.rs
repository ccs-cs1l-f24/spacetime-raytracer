use std::{fmt::Display, time::Duration};

use crate::boilerplate::BaseGpuState;

pub const TOP_OF_PHYSICS: u32 = 0;
pub const RK4_AFTER: u32 = 1;
pub const GRID_UPDATE_AFTER: u32 = 2;

pub const TOP_OF_MESHGEN: u32 = 3;

pub const NUM_QUERIES: u32 = 32;

#[derive(Clone, Debug, Default)]
#[allow(unused)] // allow unused fields :)
pub struct FramePerfStats {
    pub rk4_time: Duration,
    pub grid_update_time: Duration,
}

impl Display for FramePerfStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rk4: {:?}\nGrid Update: {:?}",
            self.rk4_time, self.grid_update_time,
        )
    }
}

pub fn get_frame_perf_stats(base: &BaseGpuState) -> FramePerfStats {
    FramePerfStats {
        rk4_time: Duration::from_nanos(
            base.query_results[RK4_AFTER as usize] - base.query_results[TOP_OF_PHYSICS as usize],
        ),
        grid_update_time: Duration::from_nanos(
            base.query_results[GRID_UPDATE_AFTER as usize] - base.query_results[RK4_AFTER as usize],
        ),
    }
}
