use std::ffi::{CStr, CString};

use raw_window_handle::RawDisplayHandle;
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuPlatform, Severity as CoreSeverity,
};
use rotex_types::{AdapterId, AdapterInfo, DebugMode, DeviceDescriptor, PlatformDescriptor};

use crate::backend::vulkan::{
    Adapter, DeviceDescriptor as BackendDeviceDescriptor, VulkanInstance,
};
use crate::bridge::VulkanBridge;
use crate::bridge::init::{init_bridge, map_device_features, map_work_queue_hints};
use crate::core::InstanceOptions;
use crate::error::{Error, ErrorKind};

/// Vulkan implementation of [`GpuPlatform`].
///
/// Prefer the backend-free helpers in `rotex-core` (`create_backend_with_selection`, etc.)
/// with a type alias at the app layer. Hold [`VulkanBridge`] directly for backend-internal access.
pub struct VulkanGpuPlatform {
    instance: VulkanInstance,
    adapters: Vec<Adapter>,
}

#[async_trait::async_trait]
impl GpuPlatform for VulkanGpuPlatform {
    type Backend = VulkanBridge;

    async fn new(platform_descriptor: PlatformDescriptor) -> Result<Self, CoreError> {
        let validation = platform_descriptor.debug == DebugMode::Validation;
        let options = InstanceOptions {
            enable_validation: validation,
            enable_debug_utils: validation,
            ..Default::default()
        };
        let mut extension_list = Vec::new();
        if let Some(display_handle) = platform_descriptor.display_handle {
            extension_list.extend(surface_extension_names(display_handle).map_err(to_core_error)?);
        }
        extension_list.sort();
        extension_list.dedup();
        let extension_names = extension_list
            .iter()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    Error::fatal(ErrorKind::Unsupported(
                        "Platform extension contains interior NUL byte",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(to_core_error)?;
        let extension_ptrs = extension_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let instance = VulkanInstance::new(&options, &extension_ptrs).map_err(to_core_error)?;
        let adapters = instance.raw().enumerate_adapters();
        Ok(Self { instance, adapters })
    }

    async fn enumerate_adapters(&self) -> Result<Vec<AdapterInfo>, CoreError> {
        let mut infos = Vec::with_capacity(self.adapters.len());
        for (index, adapter) in self.adapters.iter().enumerate() {
            let info = adapter
                .adapter_info(self.instance.raw(), AdapterId(index as u32))
                .map_err(to_core_error)?;
            infos.push(info);
        }
        Ok(infos)
    }

    async fn create_backend(
        self,
        adapter_id: AdapterId,
        device_descriptor: DeviceDescriptor,
    ) -> Result<Self::Backend, CoreError> {
        let adapter = self
            .adapters
            .get(adapter_id.0 as usize)
            .ok_or_else(|| CoreError::fatal(CoreErrorKind::NoCompatibleDevice))?;
        let enable_presentation = adapter
            .has_swapchain_extension(self.instance.raw())
            .map_err(to_core_error)?;
        let backend_desc = BackendDeviceDescriptor {
            enable_presentation,
            queues: map_work_queue_hints(device_descriptor.work_queues),
            required_features: map_device_features(device_descriptor.required_features),
        };
        let device = self
            .instance
            .request_device(adapter, backend_desc)
            .map_err(to_core_error)?;
        init_bridge(self.instance, device).map_err(to_core_error)
    }
}

fn surface_extension_names(display_handle: RawDisplayHandle) -> Result<Vec<String>, Error> {
    let names = ash_window::enumerate_required_extensions(display_handle).map_err(|_| {
        Error::fatal(ErrorKind::Unsupported(
            "surface extensions unavailable for this display",
        ))
    })?;
    Ok(names
        .iter()
        .map(|name| {
            // The pointers stay valid for the process; they address static extension names.
            let text = unsafe { CStr::from_ptr(*name) };
            text.to_string_lossy().into_owned()
        })
        .collect())
}

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        other => CoreErrorKind::Backend(format!("{other:?}")),
    };
    CoreError { kind, severity }
}
