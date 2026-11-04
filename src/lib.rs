pub mod backend;
pub mod bridge;
mod caps;
pub mod core;
pub mod error;
mod gpu_platform;

pub use backend::vulkan;
pub use bridge::VulkanBridge;
pub use error::{Error, ErrorKind, Severity};
pub use gpu_platform::VulkanGpuPlatform;
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
