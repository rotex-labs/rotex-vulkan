use std::ffi::CString;

use ash::vk;

use super::types::{DepthMode, MaterialPipelineKey, VertexLayoutId};
use super::{VulkanBridge, surface_not_attached_error};
use crate::backend::vulkan::{
    ColorBlendAttachmentState, ColorBlendState, DepthStencilState, GraphicsPipelineBuilder,
    GraphicsPipelineLayout, RasterizationState, ShaderModule, ShaderStageDescriptor,
    VertexInputDescriptor,
};
use crate::error::{Error, ErrorKind};
use rotex_types::resource::{
    BlendMode, BlendState, MaterialDescriptor, MaterialId, PrimitiveTopology, VertexBufferLayout,
    VertexFormat,
};

impl VulkanBridge {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_pipeline_for_material(
        &self,
        material: &MaterialDescriptor,
        vertex_layouts: &[VertexBufferLayout],
        dynamic_viewport: bool,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        depth_mode: DepthMode,
        topology: PrimitiveTopology,
        blend: BlendState,
    ) -> Result<super::types::MaterialPipeline, Error> {
        let vert_bytes = material.shaders.vertex.spirv_bytes().ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported(
                "Vertex shader has no SPIR-V payload",
            ))
        })?;
        let frag_bytes = material.shaders.fragment.spirv_bytes().ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported(
                "Fragment shader has no SPIR-V payload",
            ))
        })?;
        let vert_words = spv_bytes_to_words(vert_bytes);
        let frag_words = spv_bytes_to_words(frag_bytes);
        let vk_cull_mode = match material.cull_mode {
            rotex_types::CullMode::None => vk::CullModeFlags::NONE,
            rotex_types::CullMode::Front => vk::CullModeFlags::FRONT,
            rotex_types::CullMode::Back => vk::CullModeFlags::BACK,
        };
        let vertex_entry =
            CString::new(material.shaders.vertex.entry_point.as_str()).map_err(|_| {
                Error::fatal(ErrorKind::Unsupported(
                    "Vertex shader entry contains interior null byte",
                ))
            })?;
        let fragment_entry =
            CString::new(material.shaders.fragment.entry_point.as_str()).map_err(|_| {
                Error::fatal(ErrorKind::Unsupported(
                    "Fragment shader entry contains interior null byte",
                ))
            })?;
        let vert = ShaderModule::new(self.device.raw(), &vert_words)?;
        let frag = ShaderModule::new(self.device.raw(), &frag_words)?;
        let vk_layouts = super::bindings::build_material_set_layouts(
            self.device.raw(),
            &material.shaders.layout,
        )?;
        let set_layout_handles: Vec<vk::DescriptorSetLayout> =
            vk_layouts.iter().map(|l| l.handle()).collect();
        let push_ranges = super::extended::push_constant_ranges(&material.shaders.layout);
        let layout =
            GraphicsPipelineLayout::new(self.device.raw(), &set_layout_handles, &push_ranges)?;
        let pipeline = GraphicsPipelineBuilder::new()
            .with_shader_stage(
                ShaderStageDescriptor::new(vk::ShaderStageFlags::VERTEX, &vert)
                    .with_entry_name(vertex_entry.as_c_str()),
            )
            .with_shader_stage(
                ShaderStageDescriptor::new(vk::ShaderStageFlags::FRAGMENT, &frag)
                    .with_entry_name(fragment_entry.as_c_str()),
            )
            .with_input_assembly_state(
                vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vulkan_topology(topology))
                    .primitive_restart_enable(false),
            )
            .with_color_blend_state(
                ColorBlendState::default().with_attachment(blend_attachment(blend)),
            )
            .with_rasterization_state(
                RasterizationState::default()
                    .with_cull_mode(vk_cull_mode)
                    .with_front_face(vk::FrontFace::COUNTER_CLOCKWISE),
            )
            .with_depth_stencil_state(if depth_mode.is_enabled() {
                DepthStencilState::default()
                    .with_depth_test_enable(true)
                    .with_depth_write_enable(true)
                    .with_depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            } else {
                DepthStencilState::default()
            })
            .with_vertex_input_state(vertex_input_descriptor(vertex_layouts)?)
            .with_dynamic_viewport(dynamic_viewport)
            .with_render_pass(render_pass)
            .with_layout(layout.handle())
            .with_extent(extent.width, extent.height)
            .build(self.device.raw())?;
        for l in vk_layouts {
            l.destroy(self.device.raw());
        }
        vert.destroy(self.device.raw());
        frag.destroy(self.device.raw());
        Ok(super::types::MaterialPipeline { layout, pipeline })
    }

    pub(super) fn pipeline_handle_for(
        &mut self,
        material_id: MaterialId,
        vertex_layout_id: VertexLayoutId,
        depth_mode: DepthMode,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        let extent = self
            .surface_state
            .as_ref()
            .ok_or(surface_not_attached_error())?
            .swapchain
            .raw()
            .extent();
        let pipeline_key = MaterialPipelineKey {
            material_id,
            vertex_layout_id,
            depth_mode,
            render_pass,
        };

        if !self.material_pipelines.contains_key(&pipeline_key) {
            let material_desc = self
                .materials
                .get(&material_id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                .descriptor
                .clone();
            let vertex_layout = self
                .vertex_layouts
                .get(&vertex_layout_id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                .clone();
            let pipeline = self.create_pipeline_for_material(
                &material_desc,
                std::slice::from_ref(&vertex_layout),
                false,
                render_pass,
                extent,
                depth_mode,
                PrimitiveTopology::TriangleList,
                BlendState::default(),
            )?;
            self.material_pipelines.insert(pipeline_key, pipeline);
            self.pipelines_by_material
                .entry(material_id)
                .or_default()
                .insert(pipeline_key);
        }

        let pipeline = self
            .material_pipelines
            .get(&pipeline_key)
            .ok_or(Error::fatal(ErrorKind::Unsupported(
                "pipeline missing after creation",
            )))?;
        Ok((pipeline.pipeline.handle(), pipeline.layout.handle()))
    }

    pub(super) fn invalidate_material_pipelines(&mut self, material_id: MaterialId) {
        let Some(keys) = self.pipelines_by_material.remove(&material_id) else {
            return;
        };
        for key in keys {
            if let Some(pipeline) = self.material_pipelines.remove(&key) {
                pipeline.pipeline.destroy(self.device.raw());
                pipeline.layout.destroy(self.device.raw());
            }
        }
    }

    pub(super) fn destroy_all_pipelines(&mut self) {
        for (_, pipeline) in self.material_pipelines.drain() {
            pipeline.pipeline.destroy(self.device.raw());
            pipeline.layout.destroy(self.device.raw());
        }
        self.pipelines_by_material.clear();
    }
}

fn vertex_input_descriptor(layouts: &[VertexBufferLayout]) -> Result<VertexInputDescriptor, Error> {
    let mut descriptor = VertexInputDescriptor::default();
    for (binding, layout) in layouts.iter().enumerate() {
        if layout.array_stride > u32::MAX as u64 {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Vertex layout stride exceeds Vulkan limits",
            )));
        }
        let input_rate = match layout.step_mode {
            rotex_types::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
            rotex_types::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
        };
        descriptor = descriptor.with_binding(vk::VertexInputBindingDescription {
            binding: binding as u32,
            stride: layout.array_stride as u32,
            input_rate,
        });
        for attribute in &layout.attributes {
            if attribute.offset > u32::MAX as u64 {
                return Err(Error::fatal(ErrorKind::Unsupported(
                    "Vertex attribute offset exceeds Vulkan limits",
                )));
            }
            descriptor = descriptor.with_attribute(vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: binding as u32,
                format: map_vertex_format(attribute.format)?,
                offset: attribute.offset as u32,
            });
        }
    }
    Ok(descriptor)
}

fn map_vertex_format(format: VertexFormat) -> Result<vk::Format, Error> {
    let mapped = match format {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
        VertexFormat::Sint32 => vk::Format::R32_SINT,
    };
    Ok(mapped)
}

fn vulkan_topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

fn blend_attachment(blend: BlendState) -> ColorBlendAttachmentState {
    let mode = match blend.mode {
        BlendMode::Additive => BlendMode::Additive,
        BlendMode::Alpha => BlendMode::Alpha,
        BlendMode::Replace if blend.enabled => BlendMode::Alpha,
        BlendMode::Replace => BlendMode::Replace,
    };
    let mut state = ColorBlendAttachmentState::default();
    match mode {
        BlendMode::Replace => {}
        BlendMode::Additive => {
            state.blend_enable = true;
            state.src_color_blend_factor = vk::BlendFactor::ONE;
            state.dst_color_blend_factor = vk::BlendFactor::ONE;
            state.src_alpha_blend_factor = vk::BlendFactor::ONE;
            state.dst_alpha_blend_factor = vk::BlendFactor::ONE;
        }
        BlendMode::Alpha => {
            state.blend_enable = true;
            state.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
            state.dst_color_blend_factor = vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
            state.src_alpha_blend_factor = vk::BlendFactor::ONE;
            state.dst_alpha_blend_factor = vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
        }
    }
    state
}

pub(super) fn spv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().copied().map(u32::from_le_bytes).collect()
}
