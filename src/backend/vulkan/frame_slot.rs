use super::command::CommandBuffer;
use super::sync::{Fence, Semaphore};
use crate::error::Error;

/// Command buffer, fence, and image-acquired semaphore for one frame in flight.
pub struct FrameSlot {
    /// Command buffer recorded for this frame.
    pub command_buffer: CommandBuffer,
    /// Signaled when this frame's submission has finished.
    pub fence: Fence,
    /// Signaled when a swapchain image has been acquired for this frame.
    pub image_available: Semaphore,
}

impl FrameSlot {
    /// Waits for [`Self::fence`] and resets it so the slot can be recorded again.
    pub fn wait_and_reset(&self, device: &super::device::Device) -> Result<(), Error> {
        self.fence.wait(device, u64::MAX)?;
        self.fence.reset(device)
    }
}
