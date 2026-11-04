#![allow(dead_code)]
mod bindings;
mod compute_pipeline_cache;
mod extended;
pub(crate) mod init;
mod pass_targets;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::{HashMap, HashSet};

use crate::backend::vulkan::{
    CommandBuffer, CommandPool, DeferredDeleteQueue, DescriptorPoolManager, DescriptorSetLayout,
    Device, Fence, FrameSlot, VulkanDevice, VulkanInstance,
};
use crate::error::{Error, ErrorKind, vk_error};
use pass_targets::PassTargetCache;
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuBackend, Severity as CoreSeverity,
};
use rotex_types::resource::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, MaterialId, MeshId, TextureId,
    VertexBufferLayout,
};
use types::{
    BindGroupLayoutResource, BindGroupResource, BufferResource, ComputePipelineResource,
    MaterialPipeline, MaterialPipelineKey, MaterialResource, MeshResource, SurfaceState,
    TextureResource, VertexLayoutId,
};

pub struct VulkanBridge {
    instance: VulkanInstance,
    device: VulkanDevice,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    in_flight_fence: Fence,
    frame_slots: Vec<FrameSlot>,
    frames_in_flight: u32,
    current_frame_index: u32,
    current_image_index: u32,
    recording: bool,
    graphics_queue_index: u32,
    surface_state: Option<SurfaceState>,
    meshes: HashMap<MeshId, MeshResource>,
    materials: HashMap<MaterialId, MaterialResource>,
    textures: HashMap<TextureId, TextureResource>,
    bind_group_layouts: HashMap<BindGroupLayoutId, BindGroupLayoutResource>,
    bind_groups: HashMap<BindGroupId, BindGroupResource>,
    general_descriptor_pool: DescriptorPoolManager,
    storage_descriptor_pool: DescriptorPoolManager,
    empty_set_layout: DescriptorSetLayout,
    fallback_sampler: crate::backend::vulkan::RotexSampler,
    pass_target_cache: PassTargetCache,
    material_pipelines: HashMap<MaterialPipelineKey, MaterialPipeline>,
    pipelines_by_material: HashMap<MaterialId, HashSet<MaterialPipelineKey>>,
    vertex_layouts: HashMap<VertexLayoutId, VertexBufferLayout>,
    buffers: HashMap<BufferId, BufferResource>,
    compute_pipelines: HashMap<ComputePipelineId, ComputePipelineResource>,
    samplers: HashMap<rotex_types::SamplerId, crate::backend::vulkan::RotexSampler>,
    explicit_pipelines:
        HashMap<rotex_types::GraphicsPipelineId, extended::ExplicitGraphicsPipeline>,
    query_sets: HashMap<rotex_types::QuerySetId, extended::QuerySetResource>,
    buffer_states: HashMap<BufferId, rotex_types::ResourceState>,
    texture_states: HashMap<TextureId, rotex_types::ResourceState>,
    compute_pass_open: bool,
    active_compute_pipeline: Option<ComputePipelineId>,
    deferred_delete: DeferredDeleteQueue,
    active_render_pass: Option<vk::RenderPass>,
    active_pass_extent: Option<vk::Extent2D>,
    active_pipeline_layout: Option<vk::PipelineLayout>,
    active_pass: Option<rotex_types::PassDescriptor>,
    active_target_role: Option<pass_targets::TargetPassRole>,
}

use ash::vk;

fn surface_not_attached_error() -> Error {
    Error::fatal(ErrorKind::Unsupported("Surface is not attached"))
}

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        crate::error::ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        crate::error::ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        crate::error::ErrorKind::Vulkan(code) => {
            CoreErrorKind::Backend(format!("Vulkan error: {code:?} ({})", code.as_raw()))
        }
    };
    CoreError { kind, severity }
}

fn map_err_route<T>(r: Result<T, Error>) -> Result<T, CoreError> {
    r.map_err(to_core_error)
}

impl GpuBackend for VulkanBridge {
    fn capabilities(&self) -> rotex_types::BackendCapabilities {
        crate::caps::capabilities_for(crate::caps::limits_from_vk(
            &self.device.raw().properties().limits,
        ))
    }

    fn attach_surface(
        &mut self,
        surface_descriptor: rotex_types::SurfaceDescriptor,
    ) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::attach_surface(self, surface_descriptor))
    }

    fn create_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchCreate,
    ) -> Result<rotex_types::CreatedResources, CoreError> {
        map_err_route(VulkanBridge::create_resources(self, descriptor))
    }

    fn update_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchUpdate,
    ) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::update_resources(self, descriptor))
    }

    fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::execute(self, commands))
    }

    fn resize(&mut self, extent: rotex_types::Extent2D) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::resize(self, extent))
    }

    fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, CoreError> {
        map_err_route(VulkanBridge::read_texture(self, id))
    }

    fn destroy(self: Box<Self>) {
        self.destroy_all();
    }

    fn invalidate_command_cache(&mut self) {}
}

impl VulkanBridge {
    pub fn destroy_all(mut self) {
        unsafe {
            let _ = self.device.raw().logical_device().device_wait_idle();
        }
        self.deferred_delete.destroy_all(self.device.raw());
        self.destroy_all_pipelines();
        self.destroy_all_compute_pipelines();
        self.destroy_explicit_pipelines();

        let device = self.device.raw();
        for (_, buf) in self.buffers.drain() {
            buf.buffer.destroy(device);
        }
        for (_, mesh) in self.meshes.drain() {
            mesh.vertex_buffer.destroy(device);
            mesh.index_buffer.destroy(device);
        }
        for (_, tex) in self.textures.drain() {
            tex.destroy(device);
        }
        self.pass_target_cache.destroy(device);

        if let Some(state) = self.surface_state.take() {
            for sem in state.render_finished {
                sem.destroy(device);
            }
            state.image_available.destroy(device);
            state.color_targets.destroy(device);
            if let Some(dt) = state.depth_targets {
                dt.destroy(device);
            }
            state.swapchain.destroy(&self.device);
            state.surface.destroy();
        }

        for slot in self.frame_slots.drain(..) {
            slot.fence.destroy(device);
            slot.image_available.destroy(device);
        }
        self.command_pool.destroy(device);
        self.in_flight_fence.destroy(device);

        self.general_descriptor_pool.destroy(device);
        self.storage_descriptor_pool.destroy(device);

        for (_, bgl) in self.bind_group_layouts.drain() {
            bgl.layout.destroy(device);
        }
        self.empty_set_layout.destroy(device);
        self.fallback_sampler.destroy(device);

        self.device.destroy();
        self.instance.destroy();
    }

    pub fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), Error> {
        for command in commands {
            match command {
                rotex_types::RhiCommand::BeginFrame { frame_index } => {
                    let idx = *frame_index as usize % self.frame_slots.len();
                    self.current_frame_index = idx as u32;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.wait_and_reset(self.device.raw())?;
                    self.deferred_delete
                        .process_frame(self.device.raw(), self.current_frame_index as usize);
                    unsafe {
                        self.device.raw().logical_device().reset_command_buffer(
                            slot.command_buffer.handle(),
                            vk::CommandBufferResetFlags::empty(),
                        )
                    }
                    .map_err(vk_error)?;
                    slot.command_buffer.begin(
                        self.device.raw(),
                        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    )?;
                }
                rotex_types::RhiCommand::AcquirePresentationTarget => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    let sem = &slot.image_available;
                    let state = self
                        .surface_state
                        .as_ref()
                        .ok_or_else(surface_not_attached_error)?;
                    let (index, _) = state
                        .swapchain
                        .raw()
                        .acquire_next_image(sem)
                        .map_err(|_| Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    self.current_image_index = index;
                }
                rotex_types::RhiCommand::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } => {
                    let buf_res = self
                        .buffers
                        .get(buffer)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("buffer not found")))?;
                    let mapped = buf_res.buffer.map(self.device.raw())? as *mut u8;
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(mapped.add(*offset as usize), data.len())
                    };
                    target.copy_from_slice(data);
                    buf_res.buffer.unmap(self.device.raw());
                }
                rotex_types::RhiCommand::BeginRenderPass { pass, .. } => {
                    if matches!(pass.color_target, rotex_types::PassColorTarget::Texture(_)) {
                        self.begin_texture_render_pass(pass)?;
                        continue;
                    }
                    if pass.uses_depth_attachment() {
                        let _ = self.ensure_depth_targets();
                    }
                    let state = self
                        .surface_state
                        .as_ref()
                        .ok_or_else(surface_not_attached_error)?;
                    let use_depth = pass.uses_depth_attachment() && state.depth_targets.is_some();
                    let targets = if use_depth {
                        state
                            .depth_targets
                            .as_ref()
                            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                    } else {
                        &state.color_targets
                    };
                    let idx = self.current_image_index as usize;
                    if idx >= targets.framebuffers.len() {
                        return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
                    }
                    let mut clear_values = vec![vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: pass.clear_color,
                        },
                    }];
                    if use_depth {
                        clear_values.push(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: pass.clear_depth,
                                stencil: 0,
                            },
                        });
                    }
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.begin_render_pass(
                        self.device.raw(),
                        &targets.render_pass,
                        &targets.framebuffers[idx],
                        &clear_values,
                    );
                    self.active_render_pass = Some(targets.render_pass.handle());
                }
                rotex_types::RhiCommand::EndRenderPass => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.end_render_pass(self.device.raw());
                    self.active_render_pass = None;
                }
                rotex_types::RhiCommand::BindGroups {
                    first_slot,
                    bind_groups,
                    dynamic_offsets,
                } => {
                    let layout = self
                        .active_pipeline_layout
                        .ok_or(Error::fatal(ErrorKind::Unsupported("no pipeline bound")))?;
                    let mut sets = Vec::new();
                    for bg_id in bind_groups {
                        let bg = self
                            .bind_groups
                            .get(bg_id)
                            .ok_or(Error::fatal(ErrorKind::Unsupported("bind group not found")))?;
                        sets.push(bg.descriptor_set.handle());
                    }
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    if self.compute_pass_open {
                        slot.command_buffer.bind_compute_descriptor_sets(
                            self.device.raw(),
                            layout,
                            *first_slot,
                            &sets,
                        );
                    } else {
                        slot.command_buffer.bind_graphics_descriptor_sets(
                            self.device.raw(),
                            layout,
                            *first_slot,
                            &sets,
                            dynamic_offsets,
                        );
                    }
                }
                rotex_types::RhiCommand::SetVertexBuffers { mesh, .. } => {
                    let mesh_res = self
                        .meshes
                        .get(mesh)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("mesh not found")))?;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer
                        .bind_vertex_buffer(self.device.raw(), mesh_res.vertex_buffer.handle());
                }
                rotex_types::RhiCommand::SetIndexBuffer { mesh } => {
                    let mesh_res = self
                        .meshes
                        .get(mesh)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("mesh not found")))?;
                    let buf = &mesh_res.index_buffer;
                    let idx_type = mesh_res.index_type;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer
                        .bind_index_buffer(self.device.raw(), buf, 0, idx_type);
                }
                rotex_types::RhiCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                } => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.draw_indexed(
                        self.device.raw(),
                        *index_count,
                        *instance_count,
                        *first_index,
                        *vertex_offset,
                        *first_instance,
                    );
                }
                rotex_types::RhiCommand::SubmitFrame { present } => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.end(self.device.raw())?;
                    let command_buffers = [slot.command_buffer.handle()];
                    let queue = self.device.raw().get_queue(self.graphics_queue_index, 0);
                    if *present {
                        let state = self
                            .surface_state
                            .as_ref()
                            .ok_or_else(surface_not_attached_error)?;
                        let wait_sem = slot.image_available.handle();
                        let wait_semaphores = [wait_sem];
                        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                        let signal_sem =
                            state.render_finished[self.current_image_index as usize].handle();
                        let signal_semaphores = [signal_sem];
                        let submit = vk::SubmitInfo::default()
                            .wait_semaphores(&wait_semaphores)
                            .wait_dst_stage_mask(&wait_stages)
                            .command_buffers(&command_buffers)
                            .signal_semaphores(&signal_semaphores);
                        unsafe {
                            self.device.raw().logical_device().queue_submit(
                                queue,
                                &[submit],
                                slot.fence.handle(),
                            )
                        }
                        .map_err(vk_error)?;
                        let _ = state.swapchain.raw().present(
                            queue,
                            self.current_image_index,
                            &state.render_finished[self.current_image_index as usize],
                        );
                    } else {
                        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
                        unsafe {
                            self.device.raw().logical_device().queue_submit(
                                queue,
                                &[submit],
                                slot.fence.handle(),
                            )
                        }
                        .map_err(vk_error)?;
                    }
                }
                _ => self.record_phase2(command)?,
            }
        }
        Ok(())
    }

    pub fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, Error> {
        let texture = self
            .textures
            .get(&id)
            .ok_or_else(|| Error::unsupported("texture not found"))?;
        let width = texture.descriptor.width.max(1);
        let height = texture.descriptor.height.max(1);
        let format = texture.descriptor.format;
        let byte_len = format
            .expected_byte_len(width, height)
            .ok_or_else(|| Error::unsupported("texture readback size overflow"))?;
        let image = texture.image.handle();
        let staging = crate::backend::vulkan::RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            byte_len as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.submit_one_shot(|device, command_buffer| {
            command_buffer.transition_image_layout(
                device,
                image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            command_buffer.copy_image_to_buffer(
                device,
                image,
                staging.handle(),
                0,
                0,
                width,
                height,
            );
            command_buffer.transition_image_layout(
                device,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            Ok(())
        })?;
        let ptr = staging.map(self.device.raw())?;
        let mut data = vec![0_u8; byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), byte_len);
        }
        staging.unmap(self.device.raw());
        staging.destroy(self.device.raw());
        Ok(rotex_types::TextureReadback {
            data,
            format,
            width,
            height,
        })
    }

    pub(super) fn current_command_buffer(
        &self,
    ) -> Result<&crate::backend::vulkan::CommandBuffer, Error> {
        let frame_index = self.current_frame_index as usize;
        self.frame_slots
            .get(frame_index)
            .map(|slot| &slot.command_buffer)
            .ok_or_else(|| bindings::allocation_mismatch("frame slot out of range"))
    }

    pub(super) fn wait_frame_slot(&self, frame_index: u32) -> Result<(), Error> {
        let slot = self
            .frame_slots
            .get(frame_index as usize)
            .ok_or_else(|| bindings::allocation_mismatch("frame slot out of range"))?;
        slot.wait_and_reset(self.device.raw())
    }

    pub(super) fn submit_one_shot<F>(&mut self, record: F) -> Result<(), Error>
    where
        F: FnOnce(&Device, &crate::backend::vulkan::CommandBuffer) -> Result<(), Error>,
    {
        let frame_index = 0usize;
        self.wait_frame_slot(frame_index as u32)?;
        let device = self.device.raw();
        let slot = &self.frame_slots[frame_index];
        unsafe {
            device.logical_device().reset_command_buffer(
                slot.command_buffer.handle(),
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(vk_error)?;
        slot.command_buffer
            .begin(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        record(device, &slot.command_buffer)?;
        slot.command_buffer.end(device)?;
        let queue = device.get_queue(self.graphics_queue_index, 0);
        let command_buffers = [slot.command_buffer.handle()];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            device
                .logical_device()
                .queue_submit(queue, &[submit], slot.fence.handle())
        }
        .map_err(vk_error)?;
        slot.fence.wait(device, u64::MAX)?;
        Ok(())
    }
}
