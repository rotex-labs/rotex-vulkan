use ash::vk;
use rotex_core::StaticCaps;
use rotex_types::{BackendCapabilities, CoreCaps, FeatureSet, Limits, Support};

use crate::bridge::VulkanBridge;

pub(crate) const CAPS: CoreCaps = CoreCaps {
    compute: Support::Native,
    storage_buffers: Support::Native,
    push_constants: Support::Native,
    barriers: Support::Native,
    queries: Support::Native,
    copies: Support::Native,
    readback: Support::Native,
    render_to_texture: Support::Native,
    dynamic_offsets: Support::Native,
    explicit_pipelines: Support::Native,
};

impl StaticCaps for VulkanBridge {
    const CAPS: CoreCaps = CAPS;
}

pub(crate) fn limits_from_vk(limits: &vk::PhysicalDeviceLimits) -> Limits {
    Limits {
        max_texture_dimension_2d: limits.max_image_dimension2_d,
        max_bind_groups: limits.max_bound_descriptor_sets,
        max_push_constant_bytes: limits.max_push_constants_size,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_range,
        max_storage_buffer_binding_size: limits.max_storage_buffer_range,
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as u32,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment as u32,
        max_vertex_buffers: limits.max_vertex_input_bindings,
        max_vertex_attributes: limits.max_vertex_input_attributes,
        max_color_attachments: limits.max_color_attachments,
        max_compute_workgroup_size_x: limits.max_compute_work_group_size[0],
        max_compute_workgroup_size_y: limits.max_compute_work_group_size[1],
        max_compute_workgroup_size_z: limits.max_compute_work_group_size[2],
        max_compute_invocations_per_workgroup: limits.max_compute_work_group_invocations,
        max_compute_workgroups_per_dimension: limits.max_compute_work_group_count[0],
    }
}

pub(crate) fn capabilities_for(limits: Limits) -> BackendCapabilities {
    BackendCapabilities {
        core: CAPS,
        limits,
        extensions: FeatureSet::empty(),
    }
}
