use std::collections::BTreeMap;
use std::ffi::CStr;

use ash::vk;
use rotex_types::{
    AdapterCapabilities, AdapterId, AdapterInfo, AdapterQueueSupport, GpuApiBackend, HardwareFamily,
};

use crate::caps::{self, CAPS};

use crate::core::Instance;
use crate::error::{Error, ErrorKind, Severity, vk_error};

/// Kind of queue a device allocation is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueCategory {
    /// Queues that support graphics commands.
    Graphics,
    /// Queues that support compute commands.
    Compute,
    /// Queues that support transfer commands.
    Transfer,
}

/// Queues of one category to allocate from a single family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRequest {
    /// Queue capability to match.
    pub category: QueueCategory,
    /// Number of queues to take from the chosen family.
    pub count: u32,
}

/// Logical device to create from one physical device.
#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    /// Features the logical device enables.
    pub required_features: vk::PhysicalDeviceFeatures,
    /// Requests the swapchain device extension.
    pub enable_presentation: bool,
    /// Queue families to allocate.
    pub queues: Vec<QueueRequest>,
}

pub(crate) struct Adapter {
    pub(crate) handle: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    limits: vk::PhysicalDeviceLimits,
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
}

impl Adapter {
    pub(crate) fn new(
        handle: vk::PhysicalDevice,
        name: String,
        device_type: vk::PhysicalDeviceType,
        limits: vk::PhysicalDeviceLimits,
        vendor_id: u32,
        device_id: u32,
        driver_version: u32,
    ) -> Self {
        Self {
            handle,
            name,
            device_type,
            limits,
            vendor_id,
            device_id,
            driver_version,
        }
    }

    pub(crate) fn adapter_info(
        &self,
        instance: &Instance,
        id: AdapterId,
    ) -> Result<AdapterInfo, Error> {
        let supports_window_presentation = self.has_swapchain_extension(instance)?;
        let queue_support = self.queue_support(instance);
        let available_features = self.available_features(instance);

        Ok(AdapterInfo {
            id,
            name: self.name.clone(),
            hardware_family: map_hardware_family(self.device_type),
            vendor_id: self.vendor_id,
            device_id: self.device_id,
            driver_name: format_driver_version(self.driver_version),
            driver_info: String::new(),
            api_backend: GpuApiBackend::Vulkan,
            capabilities: AdapterCapabilities {
                supports_window_presentation,
                queue_support,
                features: available_features,
                limits: caps::limits_from_vk(&self.limits),
                core: CAPS,
            },
        })
    }

    fn available_features(&self, instance: &Instance) -> rotex_types::FeatureSet {
        let features = unsafe {
            instance
                .instance()
                .get_physical_device_features(self.handle)
        };
        caps::feature_set_from_vk(&features)
    }

    fn queue_support(&self, instance: &Instance) -> AdapterQueueSupport {
        let queue_families = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(self.handle)
        };
        let mut graphics_queue_count = 0;
        let mut compute_queue_count = 0;
        let mut transfer_queue_count = 0;
        for family in &queue_families {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_queue_count = graphics_queue_count.max(family.queue_count);
            }
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute_queue_count = compute_queue_count.max(family.queue_count);
            }
            if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                transfer_queue_count = transfer_queue_count.max(family.queue_count);
            }
        }
        AdapterQueueSupport {
            graphics_queue_count,
            compute_queue_count,
            transfer_queue_count,
        }
    }

    pub fn has_swapchain_extension(&self, instance: &Instance) -> Result<bool, Error> {
        let extensions = unsafe {
            instance
                .instance()
                .enumerate_device_extension_properties(self.handle)
        }
        .map_err(vk_error)?;
        Ok(extensions.iter().any(|ext| unsafe {
            CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
        }))
    }

    pub fn supports_queue_requests(&self, instance: &Instance, queues: &[QueueRequest]) -> bool {
        let queue_families = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(self.handle)
        };
        let graphics_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32);
        let compute_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32);
        let transfer_any_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(index, _)| index as u32);
        let transfer_dedicated_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(index, _)| index as u32);

        let mut has_request = false;
        for request in queues {
            if request.count == 0 {
                continue;
            }
            has_request = true;
            let family_index = match request.category {
                QueueCategory::Graphics => graphics_index,
                QueueCategory::Compute => compute_index,
                QueueCategory::Transfer => transfer_dedicated_index
                    .or(graphics_index)
                    .or(transfer_any_index)
                    .or(compute_index),
            };
            if family_index.is_none() {
                return false;
            }
        }
        has_request
    }

    pub fn request_device(
        &self,
        instance: &Instance,
        desc: DeviceDescriptor,
    ) -> Result<Device, Error> {
        let queue_families = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(self.handle)
        };
        let graphics_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32);
        let compute_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32);
        let transfer_any_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(index, _)| index as u32);
        let transfer_dedicated_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(index, _)| index as u32);

        if desc.enable_presentation {
            let extensions = unsafe {
                instance
                    .instance()
                    .enumerate_device_extension_properties(self.handle)
            }
            .map_err(vk_error)?;
            let has_swapchain = extensions.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
            });
            if !has_swapchain {
                return Err(Error {
                    kind: ErrorKind::NoCompatibleDevice,
                    severity: Severity::Fatal,
                });
            }
        }

        let mut allocations = Vec::new();
        for request in desc.queues {
            if request.count == 0 {
                continue;
            }
            let family_index = match request.category {
                QueueCategory::Graphics => graphics_index,
                QueueCategory::Compute => compute_index,
                QueueCategory::Transfer => transfer_dedicated_index
                    .or(graphics_index)
                    .or(transfer_any_index)
                    .or(compute_index),
            };
            let family_index = match family_index {
                Some(index) => index,
                None => {
                    return Err(Error {
                        kind: ErrorKind::NoCompatibleDevice,
                        severity: Severity::Fatal,
                    });
                }
            };
            allocations.push(QueueAllocation {
                category: request.category,
                family_index,
                count: request.count,
            });
        }

        if allocations.is_empty() {
            return Err(Error {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            });
        }

        let mut queue_priorities: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for allocation in &allocations {
            let entry = queue_priorities.entry(allocation.family_index).or_default();
            entry.extend(std::iter::repeat_n(1.0, allocation.count as usize));
        }

        for (family_index, priorities) in queue_priorities.iter_mut() {
            let max_supported = queue_families[*family_index as usize].queue_count as usize;
            if priorities.len() > max_supported {
                priorities.truncate(max_supported);
            }
        }

        let mut priorities_store = Vec::new();
        let mut queue_layouts = Vec::new();
        for (family_index, priorities) in queue_priorities {
            priorities_store.push(priorities);
            let idx = priorities_store.len() - 1;
            queue_layouts.push((family_index, idx));
        }

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_layouts
            .into_iter()
            .map(|(family_index, idx)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(&priorities_store[idx])
            })
            .collect();

        let device_extensions: Vec<*const i8> = if desc.enable_presentation {
            vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let mut enabled_features = desc.required_features;
        let available_features = unsafe {
            instance
                .instance()
                .get_physical_device_features(self.handle)
        };
        enabled_features.large_points = available_features.large_points;
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&enabled_features);

        let device = unsafe {
            instance
                .instance()
                .create_device(self.handle, &device_create_info, None)
        }
        .map_err(vk_error)?;

        let properties = unsafe {
            instance
                .instance()
                .get_physical_device_properties(self.handle)
        };

        Ok(Device {
            handle: self.handle,
            device,
            properties,
            queues: allocations,
        })
    }
}

/// Queues taken from one family.
#[derive(Debug, Clone)]
pub struct QueueAllocation {
    /// Category these queues serve.
    pub category: QueueCategory,
    /// Vulkan queue-family index.
    pub family_index: u32,
    /// Number of queues allocated from this family.
    pub count: u32,
}

/// Logical device, the physical device it came from, and the queues allocated with it.
pub struct Device {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    properties: vk::PhysicalDeviceProperties,
    queues: Vec<QueueAllocation>,
}

impl Device {
    /// Ash logical-device loader.
    pub fn logical_device(&self) -> &ash::Device {
        &self.device
    }

    /// Physical device handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// Properties queried when the device was created.
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    /// Queue families allocated for this device.
    pub fn queues(&self) -> &[QueueAllocation] {
        &self.queues
    }

    /// Queue `queue_index` within `family_index`.
    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> vk::Queue {
        unsafe { self.device.get_device_queue(family_index, queue_index) }
    }

    /// Returns a memory type allowed by `type_filter` that includes `properties`.
    pub fn find_memory_type(
        &self,
        instance: &Instance,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Error> {
        let memory_properties = unsafe {
            instance
                .instance()
                .get_physical_device_memory_properties(self.physical_device())
        };

        for (index, memory_type) in memory_properties.memory_types.iter().enumerate() {
            let is_allowed_by_hardware = (type_filter & (1 << index)) != 0;
            let has_required_properties = memory_type.property_flags.contains(properties);

            if is_allowed_by_hardware && has_required_properties {
                return Ok(index as u32);
            }
        }

        Err(Error::fatal(ErrorKind::NoCompatibleDevice))
    }

    /// Rounds `original_size` up to the device minimum uniform-buffer offset alignment.
    pub fn pad_uniform_buffer_size(&self, original_size: usize) -> usize {
        let min_alignment = self.properties.limits.min_uniform_buffer_offset_alignment as usize;
        let mut aligned_size = original_size;

        if min_alignment > 0 {
            aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
        }

        aligned_size
    }

    /// Destroys the logical device.
    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

fn map_hardware_family(device_type: vk::PhysicalDeviceType) -> HardwareFamily {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => HardwareFamily::DiscreteGpu,
        vk::PhysicalDeviceType::INTEGRATED_GPU => HardwareFamily::IntegratedGpu,
        vk::PhysicalDeviceType::VIRTUAL_GPU => HardwareFamily::VirtualGpu,
        vk::PhysicalDeviceType::CPU => HardwareFamily::Cpu,
        _ => HardwareFamily::Other,
    }
}

fn format_driver_version(version: u32) -> String {
    let major = (version >> 22) & 0x3ff;
    let minor = (version >> 12) & 0x3ff;
    let patch = version & 0xfff;
    format!("{major}.{minor}.{patch}")
}
