//! Optional GPU timestamps for one frame slot.
//!
//! Enabled with `ROTEX_PROFILE=1`. The bridge writes the stamps while it records
//! a frame and reads them the next time that slot's fence has signaled.

use std::time::Instant;

use ash::vk;

use crate::backend::vulkan::Device;

const STAMPS: u32 = 8;
const LAST: u32 = STAMPS - 1;

const BEGIN: u8 = 0;
const COMPUTE: u8 = 1;
const DRAW: u8 = 2;
const BOTTOM: u8 = 3;
const PAD: u8 = 4;

pub(super) struct GpuProfiler {
    pool: vk::QueryPool,
    period_ns: f64,
    frames: usize,
    next: Vec<u32>,
    ready: Vec<bool>,
    meta: Vec<[(u8, u32); STAMPS as usize]>,
    sums: [f64; STAMPS as usize],
    samples: u32,
    window_start: Instant,
}

impl GpuProfiler {
    pub(super) fn from_env(device: &Device, frames: u32) -> Option<Self> {
        std::env::var_os("ROTEX_PROFILE")?;
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(frames * STAMPS);
        let pool = unsafe {
            device
                .logical_device()
                .create_query_pool(&info, None)
                .ok()?
        };
        let frames = frames as usize;
        let empty = [(PAD, 0); STAMPS as usize];
        Some(Self {
            pool,
            period_ns: device.properties().limits.timestamp_period as f64,
            frames,
            next: vec![0; frames],
            ready: vec![false; frames],
            meta: vec![empty; frames],
            sums: [0.0; STAMPS as usize],
            samples: 0,
            window_start: Instant::now(),
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        unsafe {
            device.logical_device().destroy_query_pool(self.pool, None);
        }
    }

    pub(super) fn is_ready(&self, frame: usize) -> bool {
        frame < self.frames && self.ready[frame]
    }

    /// Folds one finished frame into the one-second average. Returns a line when the window closes.
    pub(super) fn accumulate(
        &mut self,
        frame: usize,
        ticks: &[u64; STAMPS as usize],
    ) -> Option<String> {
        if frame >= self.frames {
            return None;
        }
        let meta = self.meta[frame];
        for index in 1..STAMPS as usize {
            let delta = ticks[index].wrapping_sub(ticks[index - 1]) as f64 * self.period_ns / 1.0e6;
            self.sums[index] += delta;
        }
        self.samples += 1;
        let elapsed = self.window_start.elapsed().as_secs_f32();
        if elapsed < 1.0 || self.samples == 0 {
            return None;
        }
        let scale = 1.0 / f64::from(self.samples);
        let mut parts = Vec::new();
        for (index, stamp) in meta.iter().enumerate().skip(1) {
            let ms = self.sums[index] * scale;
            if matches!(stamp.0, BOTTOM | PAD) && ms < 0.0005 {
                continue;
            }
            let label = match stamp.0 {
                COMPUTE => "compute".to_string(),
                DRAW => format!("draw {}", stamp.1),
                BOTTOM | PAD => "tail".to_string(),
                _ => continue,
            };
            parts.push(format!("{label} {ms:.3} ms"));
        }
        self.sums = [0.0; STAMPS as usize];
        self.samples = 0;
        self.window_start = Instant::now();
        if parts.is_empty() {
            None
        } else {
            Some(format!("gpu: {}", parts.join(" | ")))
        }
    }

    /// Resets this slot's queries and writes the frame-begin stamp.
    pub(super) fn begin_slot(&mut self, frame: usize) -> Option<(vk::QueryPool, u32)> {
        if frame >= self.frames {
            return None;
        }
        self.next[frame] = 0;
        self.meta[frame] = [(PAD, 0); STAMPS as usize];
        self.note(frame, BEGIN, 0)
    }

    pub(super) fn note_compute(&mut self, frame: usize) -> Option<(vk::QueryPool, u32)> {
        self.note(frame, COMPUTE, 0)
    }

    pub(super) fn note_draw(
        &mut self,
        frame: usize,
        vertices: u32,
    ) -> Option<(vk::QueryPool, u32)> {
        self.note(frame, DRAW, vertices)
    }

    pub(super) fn finish_slot(&mut self, frame: usize) -> Vec<(vk::QueryPool, u32)> {
        let mut writes = Vec::new();
        if frame >= self.frames {
            return writes;
        }
        while self.next[frame] < LAST {
            if let Some(write) = self.note(frame, PAD, 0) {
                writes.push(write);
            } else {
                break;
            }
        }
        if let Some(write) = self.note(frame, BOTTOM, 0) {
            writes.push(write);
        }
        self.ready[frame] = true;
        writes
    }

    pub(super) fn range(&self, frame: usize) -> (vk::QueryPool, u32, u32) {
        (self.pool, (frame as u32) * STAMPS, STAMPS)
    }

    fn note(&mut self, frame: usize, kind: u8, vertices: u32) -> Option<(vk::QueryPool, u32)> {
        if frame >= self.frames || self.next[frame] >= STAMPS {
            return None;
        }
        let slot = self.next[frame];
        self.meta[frame][slot as usize] = (kind, vertices);
        self.next[frame] = slot + 1;
        Some((self.pool, (frame as u32) * STAMPS + slot))
    }
}
