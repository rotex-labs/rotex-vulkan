use std::collections::HashMap;

use ash::vk;

use super::VulkanBridge;
use crate::backend::vulkan::{
    CommandPool, DeferredDeleteQueue, DescriptorPoolManager, DescriptorSetLayout, Fence, FrameSlot,
    QueueCategory as BackendQueueCategory, QueueRequest as BackendQueueRequest, RotexSampler,
    SamplerDescriptor, VulkanDevice, VulkanInstance, general_pool_sizes, storage_pool_sizes,
};
use crate::error::{Error, ErrorKind};
use rotex_types::{
    DeviceFeatures as FrontendDeviceFeatures, Extent2D as FrontendExtent2D, WorkQueueHints,
};

const FRAMES_IN_FLIGHT: u32 = 2;

pub(crate) fn init_bridge(
    instance: VulkanInstance,
    device: VulkanDevice,
) -> Result<VulkanBridge, Error> {
    let command_pool = CommandPool::new(device.raw())?;

    let mut command_buffers = command_pool.allocate_buffers(device.raw(), 1)?;
    let command_buffer = command_buffers
        .pop()
        .ok_or_else(|| Error::fatal(ErrorKind::Unsupported("failed to allocate command buffer")))?;
    let in_flight_fence = Fence::new(device.raw(), true)?;

    let mut frame_slots = Vec::with_capacity(FRAMES_IN_FLIGHT as usize);
    for _ in 0..FRAMES_IN_FLIGHT {
        let mut buffers = command_pool.allocate_buffers(device.raw(), 1)?;
        let cb = buffers.pop().ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported("failed to allocate command buffer"))
        })?;
        let fence = Fence::new(device.raw(), true)?;
        let image_available = crate::backend::vulkan::Semaphore::new(device.raw())?;
        frame_slots.push(FrameSlot {
            command_buffer: cb,
            fence,
            image_available,
        });
    }

    let graphics_queue_index = device
        .raw()
        .queues()
        .iter()
        .find(|q| q.category == BackendQueueCategory::Graphics)
        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
        .family_index;

    let general_pool = DescriptorPoolManager::new(general_pool_sizes(), 256);
    let storage_pool = DescriptorPoolManager::new(storage_pool_sizes(), 4096);
    let empty_set_layout = DescriptorSetLayout::new(device.raw(), &[])?;
    let fallback_sampler = RotexSampler::new(
        device.raw(),
        SamplerDescriptor::default().with_filters(vk::Filter::LINEAR, vk::Filter::LINEAR),
    )?;

    Ok(VulkanBridge {
        instance,
        device,
        command_pool,
        command_buffer,
        in_flight_fence,
        frame_slots,
        frames_in_flight: FRAMES_IN_FLIGHT,
        current_frame_index: 0,
        current_image_index: 0,
        recording: false,
        graphics_queue_index,
        surface_state: None,
        meshes: HashMap::new(),
        materials: HashMap::new(),
        textures: HashMap::new(),
        bind_group_layouts: HashMap::new(),
        bind_groups: HashMap::new(),
        general_descriptor_pool: general_pool,
        storage_descriptor_pool: storage_pool,
        empty_set_layout,
        fallback_sampler,
        pass_target_cache: super::pass_targets::PassTargetCache::new(),
        material_pipelines: HashMap::new(),
        pipelines_by_material: HashMap::new(),
        vertex_layouts: HashMap::new(),
        buffers: HashMap::new(),
        compute_pipelines: HashMap::new(),
        samplers: HashMap::new(),
        explicit_pipelines: HashMap::new(),
        query_sets: HashMap::new(),
        buffer_states: HashMap::new(),
        texture_states: HashMap::new(),
        compute_pass_open: false,
        active_compute_pipeline: None,
        deferred_delete: DeferredDeleteQueue::new(),
        active_render_pass: None,
        active_pass_extent: None,
        active_pipeline_layout: None,
        active_pass: None,
        active_target_role: None,
    })
}

pub(crate) fn map_work_queue_hints(hints: WorkQueueHints) -> Vec<BackendQueueRequest> {
    let mut queues = Vec::new();
    if hints.needs_graphics {
        queues.push(BackendQueueRequest {
            category: BackendQueueCategory::Graphics,
            count: 1,
        });
    }
    if hints.needs_compute {
        queues.push(BackendQueueRequest {
            category: BackendQueueCategory::Compute,
            count: 1,
        });
    }
    if hints.needs_copy {
        queues.push(BackendQueueRequest {
            category: BackendQueueCategory::Transfer,
            count: 1,
        });
    }
    queues
}

pub(crate) fn map_device_features(features: FrontendDeviceFeatures) -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        sampler_anisotropy: features.sampler_anisotropy as u32,
        fill_mode_non_solid: features.fill_mode_non_solid as u32,
        wide_lines: features.wide_lines as u32,
        ..Default::default()
    }
}

pub(super) fn to_vk_extent(extent: FrontendExtent2D) -> vk::Extent2D {
    let extent = extent.clamped();
    vk::Extent2D {
        width: extent.width,
        height: extent.height,
    }
}
