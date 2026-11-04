//! Translation for the phase-2 commands: copies, barriers, queries, push
//! constants, explicit draws, and split compute dispatch.

use std::collections::HashMap;

use ash::vk::{self, Handle};

use super::VulkanBridge;
use super::types::{BufferResource, TextureResource};
use crate::backend::vulkan::RotexSampler;
use crate::error::Error;
use rotex_types::{
    BarrierResource, BufferId, BufferUsages, GraphicsPipelineDescriptor, GraphicsPipelineId,
    QuerySetId, QueryType, ResourceState, TextureId,
};

pub(super) struct QuerySetResource {
    pub(super) pool: vk::QueryPool,
    pub(super) count: u32,
    pub(super) ty: QueryType,
    pub(super) reset_recorded: bool,
}

pub(super) struct ExplicitGraphicsPipeline {
    pub(super) descriptor: GraphicsPipelineDescriptor,
    pub(super) compiled: HashMap<u64, (vk::Pipeline, vk::PipelineLayout)>,
}

impl VulkanBridge {
    pub(super) fn record_phase2(&mut self, command: &rotex_types::RhiCommand) -> Result<(), Error> {
        match command {
            rotex_types::RhiCommand::BeginComputePass { .. } => {
                self.compute_pass_open = true;
            }
            rotex_types::RhiCommand::EndComputePass => {
                self.compute_pass_open = false;
                self.active_compute_pipeline = None;
            }
            rotex_types::RhiCommand::SetComputePipeline { pipeline } => {
                self.bind_compute_pipeline_command(*pipeline)?;
            }
            rotex_types::RhiCommand::Dispatch { workgroup_count } => {
                if self.active_compute_pipeline.is_none() {
                    return Err(Error::unsupported("no compute pipeline bound"));
                }
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer.dispatch(
                    device,
                    workgroup_count[0],
                    workgroup_count[1],
                    workgroup_count[2],
                );
            }
            rotex_types::RhiCommand::SetViewport {
                x,
                y,
                width,
                height,
                min_depth,
                max_depth,
            } => {
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer.set_viewport(
                    device,
                    vk::Viewport {
                        x: *x,
                        y: *y,
                        width: *width,
                        height: *height,
                        min_depth: *min_depth,
                        max_depth: *max_depth,
                    },
                );
            }
            rotex_types::RhiCommand::SetScissor {
                x,
                y,
                width,
                height,
            } => {
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer.set_scissor(
                    device,
                    vk::Rect2D {
                        offset: vk::Offset2D {
                            x: *x as i32,
                            y: *y as i32,
                        },
                        extent: vk::Extent2D {
                            width: *width,
                            height: *height,
                        },
                    },
                );
            }
            rotex_types::RhiCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer.draw_vertices(
                    device,
                    *vertex_count,
                    *instance_count,
                    *first_vertex,
                    *first_instance,
                );
            }
            rotex_types::RhiCommand::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => self.copy_buffers(*src, *src_offset, *dst, *dst_offset, *size)?,
            rotex_types::RhiCommand::CopyBufferToTexture {
                src,
                src_offset,
                dst,
                bytes_per_row,
                width,
                height,
            } => self.copy_buffer_to_texture(
                *src,
                *src_offset,
                *dst,
                *bytes_per_row,
                *width,
                *height,
            )?,
            rotex_types::RhiCommand::CopyTextureToBuffer {
                src,
                dst,
                dst_offset,
                bytes_per_row,
                width,
                height,
            } => self.copy_texture_to_buffer(
                *src,
                *dst,
                *dst_offset,
                *bytes_per_row,
                *width,
                *height,
            )?,
            rotex_types::RhiCommand::Barrier { resource, state } => {
                self.record_barrier(*resource, *state)?;
            }
            rotex_types::RhiCommand::PushConstants {
                stages,
                offset,
                data,
            } => self.record_push_constants(*stages, *offset, data)?,
            rotex_types::RhiCommand::WriteTimestamp { query_set, index } => {
                self.record_timestamp(*query_set, *index)?;
            }
            rotex_types::RhiCommand::ResolveQuerySet {
                query_set,
                first_query,
                query_count,
                destination,
                offset,
            } => self.resolve_queries(
                *query_set,
                *first_query,
                *query_count,
                *destination,
                *offset,
            )?,
            rotex_types::RhiCommand::SetGraphicsPipeline { pipeline } => {
                self.bind_explicit_graphics_pipeline(*pipeline)?;
            }
            rotex_types::RhiCommand::BindVertexBuffer {
                buffer,
                binding,
                offset,
            } => {
                let handle = self
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| Error::unsupported("vertex buffer not found"))?
                    .buffer
                    .handle();
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer
                    .bind_vertex_buffer_at(device, *binding, handle, *offset);
            }
            rotex_types::RhiCommand::BindIndexBuffer {
                buffer,
                offset,
                format,
            } => {
                let buf = self
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| Error::unsupported("index buffer not found"))?;
                let index_type = super::resources::map_index_type(*format);
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer
                    .bind_index_buffer(device, &buf.buffer, *offset, index_type);
            }
            rotex_types::RhiCommand::Extension {
                feature,
                opcode,
                payload,
            } => {
                use rotex_core::Feature;
                use rotex_types::IndirectDraw;
                if *feature != IndirectDraw::ID || *opcode != 1 || payload.len() < 16 {
                    return Err(Error::unsupported("extension command is not implemented"));
                }
                let mut buffer_bytes = [0u8; 8];
                let mut offset_bytes = [0u8; 8];
                buffer_bytes.copy_from_slice(&payload[0..8]);
                offset_bytes.copy_from_slice(&payload[8..16]);
                let buffer_id = BufferId(u64::from_le_bytes(buffer_bytes));
                let offset = u64::from_le_bytes(offset_bytes);
                let buffer = self
                    .buffers
                    .get(&buffer_id)
                    .ok_or_else(|| Error::unsupported("indirect buffer not found"))?;
                let handle = buffer.buffer.handle();
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer
                    .draw_indexed_indirect(device, handle, offset);
            }
            _ => return Err(Error::unsupported("command not implemented")),
        }
        Ok(())
    }

    pub(super) fn note_buffer_created(&mut self, id: BufferId, usages: BufferUsages) {
        let state = if usages.contains(BufferUsages::VERTEX) {
            ResourceState::VertexBuffer
        } else if usages.contains(BufferUsages::INDEX) {
            ResourceState::IndexBuffer
        } else if usages.contains(BufferUsages::UNIFORM) {
            ResourceState::Uniform
        } else if usages.contains(BufferUsages::STORAGE) {
            ResourceState::ShaderWrite
        } else {
            ResourceState::CopyDst
        };
        self.buffer_states.insert(id, state);
    }

    pub(super) fn note_texture_created(&mut self, id: TextureId, resource: &TextureResource) {
        let aspects = resource.descriptor.format.info().aspects;
        let state = if aspects.contains(rotex_types::FormatAspects::DEPTH) {
            ResourceState::DepthTarget
        } else if resource.descriptor.render_attachment {
            ResourceState::ColorTarget
        } else {
            ResourceState::ShaderRead
        };
        self.texture_states.insert(id, state);
    }

    fn bind_compute_pipeline_command(
        &mut self,
        pipeline: rotex_types::ComputePipelineId,
    ) -> Result<(), Error> {
        let (handle, layout) = {
            let resource = self
                .compute_pipelines
                .get(&pipeline)
                .ok_or_else(|| Error::unsupported("compute pipeline not found"))?;
            (
                resource.pipeline.handle(),
                resource.pipeline_layout.handle(),
            )
        };
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer.bind_compute_pipeline(device, handle);
        self.active_pipeline_layout = Some(layout);
        self.active_compute_pipeline = Some(pipeline);
        Ok(())
    }

    fn copy_buffers(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: BufferId,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let src_buffer = self
            .buffers
            .get(&src)
            .ok_or_else(|| Error::unsupported("copy source buffer not found"))?
            .buffer
            .handle();
        let dst_buffer = self
            .buffers
            .get(&dst)
            .ok_or_else(|| Error::unsupported("copy destination buffer not found"))?
            .buffer
            .handle();
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer
            .copy_buffer_region(device, src_buffer, src_offset, dst_buffer, dst_offset, size);
        Ok(())
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: TextureId,
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        let buffer = self
            .buffers
            .get(&src)
            .ok_or_else(|| Error::unsupported("copy source buffer not found"))?
            .buffer
            .handle();
        let (image, bpp) = {
            let texture = self
                .textures
                .get(&dst)
                .ok_or_else(|| Error::unsupported("copy destination texture not found"))?;
            (
                texture.image.handle(),
                texture.descriptor.format.info().block_bytes.max(1),
            )
        };
        let row_length = if bytes_per_row == 0 {
            0
        } else {
            bytes_per_row / bpp
        };
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer
            .copy_buffer_to_image_at(device, buffer, src_offset, row_length, image, width, height);
        Ok(())
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: TextureId,
        dst: BufferId,
        dst_offset: u64,
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        let (image, bpp) = {
            let texture = self
                .textures
                .get(&src)
                .ok_or_else(|| Error::unsupported("copy source texture not found"))?;
            (
                texture.image.handle(),
                texture.descriptor.format.info().block_bytes.max(1),
            )
        };
        let buffer = self
            .buffers
            .get(&dst)
            .ok_or_else(|| Error::unsupported("copy destination buffer not found"))?
            .buffer
            .handle();
        let row_length = if bytes_per_row == 0 {
            0
        } else {
            bytes_per_row / bpp
        };
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer
            .copy_image_to_buffer(device, image, buffer, dst_offset, row_length, width, height);
        Ok(())
    }

    fn record_barrier(
        &mut self,
        resource: BarrierResource,
        state: ResourceState,
    ) -> Result<(), Error> {
        match resource {
            BarrierResource::Buffer(id) => {
                let previous = self
                    .buffer_states
                    .insert(id, state)
                    .unwrap_or(ResourceState::CopyDst);
                let buffer = self
                    .buffers
                    .get(&id)
                    .ok_or_else(|| Error::unsupported("barrier buffer not found"))?;
                let size = buffer.size;
                let handle = buffer.buffer.handle();
                let (src_access, src_stage) = buffer_access(previous);
                let (dst_access, dst_stage) = buffer_access(state);
                let device = self.device.raw();
                let slot = &self.frame_slots[self.current_frame_index as usize];
                slot.command_buffer.buffer_barrier(
                    device, handle, size, src_access, dst_access, src_stage, dst_stage,
                );
            }
            BarrierResource::Texture(id) => {
                let _previous = self
                    .texture_states
                    .insert(id, state)
                    .unwrap_or(ResourceState::ShaderRead);
                let new_layout = image_layout(state);
                let frame = self.current_frame_index as usize;
                let device = self.device.raw();
                let texture = self
                    .textures
                    .get(&id)
                    .ok_or_else(|| Error::unsupported("barrier texture not found"))?;
                let command_buffer = &self.frame_slots[frame].command_buffer;
                texture
                    .image
                    .transition_layout(device, command_buffer, new_layout);
            }
        }
        Ok(())
    }

    fn record_push_constants(
        &mut self,
        stages: rotex_types::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) -> Result<(), Error> {
        let layout = self
            .active_pipeline_layout
            .ok_or_else(|| Error::unsupported("no pipeline bound for push constants"))?;
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer.push_constants(
            device,
            layout,
            super::bindings::map_shader_stages(stages),
            offset,
            data,
        );
        Ok(())
    }

    fn record_timestamp(&mut self, query_set: QuerySetId, index: u32) -> Result<(), Error> {
        let (pool, count, ty, needs_reset) = {
            let resource = self
                .query_sets
                .get(&query_set)
                .ok_or_else(|| Error::unsupported("query set not found"))?;
            (
                resource.pool,
                resource.count,
                resource.ty,
                !resource.reset_recorded,
            )
        };
        if needs_reset {
            if self.active_render_pass.is_some() {
                return Err(Error::unsupported(
                    "query pool must be reset outside a render pass",
                ));
            }
            let device = self.device.raw();
            let slot = &self.frame_slots[self.current_frame_index as usize];
            slot.command_buffer.reset_queries(device, pool, count);
            if let Some(resource) = self.query_sets.get_mut(&query_set) {
                resource.reset_recorded = true;
            }
        }
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        match ty {
            QueryType::Timestamp => {
                slot.command_buffer.write_timestamp(device, pool, index);
            }
            QueryType::Occlusion => {
                slot.command_buffer.begin_query(device, pool, index);
                slot.command_buffer.end_query(device, pool, index);
            }
        }
        Ok(())
    }

    fn resolve_queries(
        &mut self,
        query_set: QuerySetId,
        first: u32,
        count: u32,
        destination: BufferId,
        offset: u64,
    ) -> Result<(), Error> {
        let pool = self
            .query_sets
            .get(&query_set)
            .ok_or_else(|| Error::unsupported("query set not found"))?
            .pool;
        let buffer = self
            .buffers
            .get(&destination)
            .ok_or_else(|| Error::unsupported("query destination buffer not found"))?
            .buffer
            .handle();
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer
            .resolve_queries(device, pool, first, count, buffer, offset);
        Ok(())
    }

    fn bind_explicit_graphics_pipeline(
        &mut self,
        pipeline: GraphicsPipelineId,
    ) -> Result<(), Error> {
        let render_pass = self
            .active_render_pass
            .ok_or_else(|| Error::unsupported("no active render pass"))?;
        let key = render_pass.as_raw();
        if !self
            .explicit_pipelines
            .get(&pipeline)
            .ok_or_else(|| Error::unsupported("graphics pipeline not found"))?
            .compiled
            .contains_key(&key)
        {
            self.compile_explicit_pipeline(pipeline, render_pass)?;
        }
        let compiled = self
            .explicit_pipelines
            .get(&pipeline)
            .ok_or_else(|| Error::unsupported("graphics pipeline not found"))?
            .compiled
            .get(&key)
            .copied()
            .ok_or_else(|| Error::unsupported("graphics pipeline not compiled"))?;
        let device = self.device.raw();
        let slot = &self.frame_slots[self.current_frame_index as usize];
        slot.command_buffer
            .bind_graphics_pipeline(device, compiled.0);
        if let Some(extent) = self.active_pass_extent {
            let slot = &self.frame_slots[self.current_frame_index as usize];
            slot.command_buffer.set_viewport(
                device,
                vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
            );
            slot.command_buffer.set_scissor(
                device,
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
            );
        }
        self.active_pipeline_layout = Some(compiled.1);
        self.active_compute_pipeline = None;
        Ok(())
    }

    fn compile_explicit_pipeline(
        &mut self,
        pipeline: GraphicsPipelineId,
        render_pass: vk::RenderPass,
    ) -> Result<(), Error> {
        let descriptor = self
            .explicit_pipelines
            .get(&pipeline)
            .ok_or_else(|| Error::unsupported("graphics pipeline not found"))?
            .descriptor
            .clone();
        let vertex_layouts = descriptor.vertex_buffers.clone();
        let mut material = rotex_types::MaterialDescriptor::new(
            descriptor.shaders.clone(),
            descriptor.depth_enabled,
            None,
        )
        .with_cull_mode(descriptor.cull_mode);
        material.enable_depth = descriptor.depth_enabled;
        let extent = self.active_pass_extent.unwrap_or(vk::Extent2D {
            width: 1,
            height: 1,
        });
        let depth_mode = if descriptor.depth_enabled {
            super::types::DepthMode::Enabled
        } else {
            super::types::DepthMode::Disabled
        };
        let compiled = self.create_pipeline_for_material(
            &material,
            &vertex_layouts,
            true,
            render_pass,
            extent,
            depth_mode,
        )?;
        let handles = (compiled.pipeline.handle(), compiled.layout.handle());
        self.explicit_pipelines
            .get_mut(&pipeline)
            .ok_or_else(|| Error::unsupported("graphics pipeline not found"))?
            .compiled
            .insert(render_pass.as_raw(), handles);
        Ok(())
    }

    pub(super) fn destroy_explicit_pipelines(&mut self) {
        let device = self.device.raw();
        for (_, pipeline) in self.explicit_pipelines.drain() {
            for (_, (pipeline_handle, layout_handle)) in pipeline.compiled {
                unsafe {
                    device
                        .logical_device()
                        .destroy_pipeline(pipeline_handle, None);
                    device
                        .logical_device()
                        .destroy_pipeline_layout(layout_handle, None);
                }
            }
        }
        for (_, sampler) in self.samplers.drain() {
            sampler.destroy(device);
        }
        for (_, query) in self.query_sets.drain() {
            unsafe {
                device.logical_device().destroy_query_pool(query.pool, None);
            }
        }
    }

    pub(super) fn create_sampler_resource(
        &self,
        descriptor: &rotex_types::SamplerDescriptor,
    ) -> Result<RotexSampler, Error> {
        let mag_filter = match descriptor.mag_filter {
            rotex_types::FilterMode::Nearest => vk::Filter::NEAREST,
            rotex_types::FilterMode::Linear => vk::Filter::LINEAR,
        };
        let min_filter = match descriptor.min_filter {
            rotex_types::FilterMode::Nearest => vk::Filter::NEAREST,
            rotex_types::FilterMode::Linear => vk::Filter::LINEAR,
        };
        let address = match descriptor.address_mode {
            rotex_types::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            rotex_types::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            rotex_types::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };
        let sampler = crate::backend::vulkan::SamplerDescriptor {
            mag_filter,
            min_filter,
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            ..Default::default()
        };
        RotexSampler::new(self.device.raw(), sampler)
    }

    pub(super) fn create_query_pool(
        &self,
        count: u32,
        ty: QueryType,
    ) -> Result<vk::QueryPool, Error> {
        let query_type = match ty {
            QueryType::Timestamp => vk::QueryType::TIMESTAMP,
            QueryType::Occlusion => vk::QueryType::OCCLUSION,
        };
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(query_type)
            .query_count(count.max(1));
        unsafe {
            self.device
                .raw()
                .logical_device()
                .create_query_pool(&info, None)
        }
        .map_err(crate::error::vk_error)
    }
}

pub(super) fn buffer_access(state: ResourceState) -> (vk::AccessFlags, vk::PipelineStageFlags) {
    match state {
        ResourceState::CopySrc => (
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        ),
        ResourceState::CopyDst => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        ResourceState::ShaderRead | ResourceState::Uniform => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        ResourceState::ShaderWrite => (
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        ResourceState::VertexBuffer => (
            vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            vk::PipelineStageFlags::VERTEX_INPUT,
        ),
        ResourceState::IndexBuffer => (
            vk::AccessFlags::INDEX_READ,
            vk::PipelineStageFlags::VERTEX_INPUT,
        ),
        ResourceState::IndirectArg => (
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags::DRAW_INDIRECT,
        ),
        ResourceState::ColorTarget => (
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        ResourceState::DepthTarget => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        ResourceState::Present => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ),
        _ => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    }
}

pub(super) fn image_layout(state: ResourceState) -> vk::ImageLayout {
    match state {
        ResourceState::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ResourceState::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ResourceState::ColorTarget => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ResourceState::DepthTarget => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ResourceState::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        ResourceState::ShaderRead | ResourceState::ShaderWrite => {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
        _ => vk::ImageLayout::GENERAL,
    }
}

pub(super) fn push_constant_ranges(
    layout: &rotex_types::AbstractPipelineLayout,
) -> Vec<vk::PushConstantRange> {
    layout
        .push_constants
        .iter()
        .map(|range| vk::PushConstantRange {
            stage_flags: super::bindings::map_shader_stages(range.stages),
            offset: range.offset,
            size: range.size,
        })
        .collect()
}

impl BufferResource {
    pub(super) fn whole_size(&self) -> vk::DeviceSize {
        self.size
    }
}
