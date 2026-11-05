use ash::vk;

use super::device::Device;
use crate::error::{Error, vk_error};

/// Binary semaphore.
pub struct Semaphore {
    pub(crate) handle: vk::Semaphore,
}

impl Semaphore {
    /// Creates an unsignaled binary semaphore.
    pub fn new(device: &Device) -> Result<Self, Error> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let handle = unsafe { device.logical_device().create_semaphore(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// Vulkan semaphore handle.
    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }

    /// Destroys the semaphore.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_semaphore(self.handle, None);
        }
    }
}

/// Fence that can start signaled.
pub struct Fence {
    pub(crate) handle: vk::Fence,
}

impl Fence {
    /// Creates a fence, already signaled when `signaled` is set.
    pub fn new(device: &Device, signaled: bool) -> Result<Self, Error> {
        let mut create_info = vk::FenceCreateInfo::default();

        if signaled {
            create_info = create_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        let handle = unsafe { device.logical_device().create_fence(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// Vulkan fence handle.
    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    /// Waits until the fence is signaled or `timeout_ns` elapses.
    pub fn wait(&self, device: &Device, timeout_ns: u64) -> Result<(), Error> {
        unsafe {
            device
                .logical_device()
                .wait_for_fences(&[self.handle], true, timeout_ns)
        }
        .map_err(vk_error)
    }

    /// Returns the fence to the unsignaled state.
    pub fn reset(&self, device: &Device) -> Result<(), Error> {
        unsafe { device.logical_device().reset_fences(&[self.handle]) }.map_err(vk_error)
    }

    /// Destroys the fence.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_fence(self.handle, None);
        }
    }
}
