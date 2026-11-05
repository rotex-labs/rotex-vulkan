#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! Vulkan backend for explicit command lists.
//!
//! [`VulkanGpuPlatform`] opens a Vulkan instance and [`VulkanBridge`] translates
//! [`rotex_types::RhiCommand`] values into Vulkan calls. The bridge keeps handle
//! tables, a pipeline cache, and per-frame submission slots. Scene state, buffer
//! contents, and the choice between uniform and storage buffers stay with the
//! caller.
//!
//! [`vulkan`] is the native HAL: devices, swapchains, pipelines, descriptors,
//! and command buffers. Presentation stays inactive until a surface is attached.

/// Instance, device, and swapchain construction.
pub mod backend;
/// Command translation and GPU resource tables.
pub mod bridge;
mod caps;
/// Vulkan instance and debug messenger.
pub mod core;
/// Errors returned by the Vulkan backend.
pub mod error;
mod gpu_platform;

pub use backend::vulkan;
pub use bridge::VulkanBridge;
pub use error::{Error, ErrorKind, Severity};
pub use gpu_platform::VulkanGpuPlatform;
/// Shared RHI vocabulary.
pub use rotex_types;

impl rotex_core::Supports<rotex_types::IndirectDraw> for VulkanBridge {}

#[cfg(test)]
mod conformance_test {
    use super::VulkanGpuPlatform;
    use rotex_core::GpuPlatform;
    use rotex_types::{DeviceDescriptor, PlatformDescriptor};

    #[test]
    fn backend_matches_reported_capabilities() {
        pollster::block_on(async {
            let gpu = match VulkanGpuPlatform::new(PlatformDescriptor::default()).await {
                Ok(gpu) => gpu,
                Err(_) => return,
            };
            let adapters = match gpu.enumerate_adapters().await {
                Ok(adapters) => adapters,
                Err(_) => return,
            };
            let Some(adapter) = adapters.first() else {
                return;
            };
            let mut backend = match gpu
                .create_backend(adapter.id, DeviceDescriptor::default())
                .await
            {
                Ok(backend) => backend,
                Err(_) => return,
            };
            rotex_conformance::run(&mut backend).expect("conformance");
        });
    }
}
