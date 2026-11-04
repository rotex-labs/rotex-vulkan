use ash::vk;

use super::buffer::RotexBuffer;
use super::device::{Device, QueueCategory};
use super::framebuffer::Framebuffer;
use super::pass::RenderPass;
use crate::error::{Error, ErrorKind, Severity, vk_error};

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn begin(&self, device: &Device, flags: vk::CommandBufferUsageFlags) -> Result<(), Error> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);

        unsafe {
            device
                .logical_device()
                .begin_command_buffer(self.handle, &begin_info)
        }
        .map_err(vk_error)
    }

    pub fn begin_render_pass(
        &self,
        device: &Device,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_values: &[vk::ClearValue],
    ) {
        debug_assert_eq!(
            clear_values.len() as u32,
            render_pass.attachments().len() as u32,
            "Rotex Core Panic: The number of clear values does not match the Render Pass attachment count!"
        );

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: framebuffer.extent(),
            })
            .clear_values(clear_values);

        unsafe {
            device.logical_device().cmd_begin_render_pass(
                self.handle,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_render_pass(&self, device: &Device) {
        unsafe {
            device.logical_device().cmd_end_render_pass(self.handle);
        }
    }

    pub fn end(&self, device: &Device) -> Result<(), Error> {
        unsafe { device.logical_device().end_command_buffer(self.handle) }.map_err(vk_error)
    }

    pub fn bind_graphics_pipeline(&self, device: &Device, pipeline: vk::Pipeline) {
        unsafe {
            device.logical_device().cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }
    }

    pub fn bind_graphics_descriptor_sets(
        &self,
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            device.logical_device().cmd_bind_descriptor_sets(
                self.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    pub fn bind_compute_pipeline(&self, device: &Device, pipeline: vk::Pipeline) {
        unsafe {
            device.logical_device().cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        }
    }

    pub fn bind_compute_descriptor_sets(
        &self,
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            device.logical_device().cmd_bind_descriptor_sets(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                first_set,
                descriptor_sets,
                &[],
            );
        }
    }

    pub fn dispatch(&self, device: &Device, x: u32, y: u32, z: u32) {
        unsafe {
            device.logical_device().cmd_dispatch(self.handle, x, y, z);
        }
    }

    pub fn bind_vertex_buffer(&self, device: &Device, buffer: vk::Buffer) {
        self.bind_vertex_buffer_at(device, 0, buffer, 0);
    }

    pub fn bind_vertex_buffer_at(
        &self,
        device: &Device,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        unsafe {
            device.logical_device().cmd_bind_vertex_buffers(
                self.handle,
                binding,
                &[buffer],
                &[offset],
            );
        }
    }

    pub fn draw(&self, device: &Device, vertex_count: u32) {
        unsafe {
            device
                .logical_device()
                .cmd_draw(self.handle, vertex_count, 1, 0, 0);
        }
    }

    pub fn set_viewport(&self, device: &Device, viewport: vk::Viewport) {
        unsafe {
            device
                .logical_device()
                .cmd_set_viewport(self.handle, 0, &[viewport]);
        }
    }

    pub fn set_scissor(&self, device: &Device, scissor: vk::Rect2D) {
        unsafe {
            device
                .logical_device()
                .cmd_set_scissor(self.handle, 0, &[scissor]);
        }
    }

    pub fn transition_image_layout(
        &self,
        device: &Device,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let (src_access, src_stage) = Self::infer_state(old_layout);
        let (dst_access, dst_stage) = Self::infer_state(new_layout);

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);

        unsafe {
            device.logical_device().cmd_pipeline_barrier(
                self.handle,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn copy_buffer_to_image(
        &self,
        device: &Device,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            device.logical_device().cmd_copy_buffer_to_image(
                self.handle,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    pub fn copy_buffer(
        &self,
        device: &Device,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        self.copy_buffer_region(device, src, 0, dst, 0, size);
    }

    pub fn copy_buffer_region(
        &self,
        device: &Device,
        src: vk::Buffer,
        src_offset: vk::DeviceSize,
        dst: vk::Buffer,
        dst_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        unsafe {
            device
                .logical_device()
                .cmd_copy_buffer(self.handle, src, dst, &[region]);
        }
    }

    pub fn copy_buffer_to_image_at(
        &self,
        device: &Device,
        buffer: vk::Buffer,
        buffer_offset: vk::DeviceSize,
        row_length: u32,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(row_length)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            device.logical_device().cmd_copy_buffer_to_image(
                self.handle,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    pub fn copy_image_to_buffer(
        &self,
        device: &Device,
        image: vk::Image,
        buffer: vk::Buffer,
        buffer_offset: vk::DeviceSize,
        row_length: u32,
        width: u32,
        height: u32,
    ) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(row_length)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            device.logical_device().cmd_copy_image_to_buffer(
                self.handle,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
        }
    }

    pub fn buffer_barrier(
        &self,
        device: &Device,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(0)
            .size(size);
        unsafe {
            device.logical_device().cmd_pipeline_barrier(
                self.handle,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    pub fn push_constants(
        &self,
        device: &Device,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            device
                .logical_device()
                .cmd_push_constants(self.handle, layout, stages, offset, data);
        }
    }

    pub fn draw_indexed_indirect(
        &self,
        device: &Device,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        unsafe {
            device.logical_device().cmd_draw_indexed_indirect(
                self.handle,
                buffer,
                offset,
                1,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
    }

    pub fn draw_vertices(
        &self,
        device: &Device,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            device.logical_device().cmd_draw(
                self.handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn reset_queries(&self, device: &Device, pool: vk::QueryPool, count: u32) {
        unsafe {
            device
                .logical_device()
                .cmd_reset_query_pool(self.handle, pool, 0, count);
        }
    }

    pub fn write_timestamp(&self, device: &Device, pool: vk::QueryPool, index: u32) {
        unsafe {
            device.logical_device().cmd_write_timestamp(
                self.handle,
                vk::PipelineStageFlags::ALL_COMMANDS,
                pool,
                index,
            );
        }
    }

    pub fn begin_query(&self, device: &Device, pool: vk::QueryPool, index: u32) {
        unsafe {
            device.logical_device().cmd_begin_query(
                self.handle,
                pool,
                index,
                vk::QueryControlFlags::empty(),
            );
        }
    }

    pub fn end_query(&self, device: &Device, pool: vk::QueryPool, index: u32) {
        unsafe {
            device
                .logical_device()
                .cmd_end_query(self.handle, pool, index);
        }
    }

    pub fn resolve_queries(
        &self,
        device: &Device,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
    ) {
        unsafe {
            device.logical_device().cmd_copy_query_pool_results(
                self.handle,
                pool,
                first,
                count,
                buffer,
                offset,
                std::mem::size_of::<u64>() as vk::DeviceSize,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            );
        }
    }

    fn infer_state(layout: vk::ImageLayout) -> (vk::AccessFlags, vk::PipelineStageFlags) {
        match layout {
            vk::ImageLayout::UNDEFINED => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            _ => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
        }
    }

    pub fn bind_index_buffer(
        &self,
        device: &Device,
        buffer: &RotexBuffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            device.logical_device().cmd_bind_index_buffer(
                self.handle,
                buffer.handle(),
                offset,
                index_type,
            );
        }
    }

    pub fn draw_indexed(
        &self,
        device: &Device,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            device.logical_device().cmd_draw_indexed(
                self.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
}

pub struct CommandPool {
    pub(crate) handle: vk::CommandPool,
}

impl CommandPool {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let graphics_queue = device
            .queues()
            .iter()
            .find(|q| q.category == QueueCategory::Graphics)
            .ok_or(Error {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            })?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue.family_index);

        let handle = unsafe {
            device
                .logical_device()
                .create_command_pool(&pool_info, None)
        }
        .map_err(vk_error)?;

        Ok(Self { handle })
    }

    pub fn allocate_buffers(
        &self,
        device: &Device,
        count: u32,
    ) -> Result<Vec<CommandBuffer>, Error> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let handles = unsafe {
            device
                .logical_device()
                .allocate_command_buffers(&alloc_info)
        }
        .map_err(vk_error)?;

        Ok(handles
            .into_iter()
            .map(|handle| CommandBuffer { handle })
            .collect())
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_command_pool(self.handle, None);
        }
    }
}
