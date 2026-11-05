use ash::vk;

/// Viewport rectangle and depth range.
pub struct Viewport {
    /// Left edge, in pixels.
    pub x: f32,
    /// Top edge, in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Near end of the depth range.
    pub min_depth: f32,
    /// Far end of the depth range.
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0,
            height: 0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl Viewport {
    /// Sets the top-left corner in pixels.
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Sets the size in pixels.
    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the depth range.
    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

    /// Converts this viewport to Vulkan viewport state.
    pub fn to_vk_viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: self.x,
            y: self.y,
            width: self.width as f32,
            height: self.height as f32,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

/// Rasterizer state for one graphics pipeline.
pub struct RasterizationState {
    /// Clamps fragments to the depth range instead of clipping them.
    pub depth_clamp_enable: bool,
    /// Discards primitives before rasterization.
    pub rasterizer_discard_enable: bool,
    /// Fill, line, or point mode.
    pub polygon_mode: vk::PolygonMode,
    /// Faces that are discarded.
    pub cull_mode: vk::CullModeFlags,
    /// Winding that counts as a front face.
    pub front_face: vk::FrontFace,
    /// Enables depth bias.
    pub depth_bias_enable: bool,
    /// Constant depth-bias factor.
    pub depth_bias_constant_factor: f32,
    /// Maximum absolute depth bias.
    pub depth_bias_clamp: f32,
    /// Slope-scaled depth-bias factor.
    pub depth_bias_slope_factor: f32,
    /// Rasterization create flags.
    pub flags: vk::PipelineRasterizationStateCreateFlags,
    /// Width of rasterized lines, in pixels.
    pub line_width: f32,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            line_width: 1.0,
        }
    }
}

impl RasterizationState {
    /// Enables or disables depth clamp.
    pub fn with_depth_clamp_enable(mut self, enable: bool) -> Self {
        self.depth_clamp_enable = enable;
        self
    }

    /// Enables or disables discarding primitives before rasterization.
    pub fn with_rasterizer_discard_enable(mut self, enable: bool) -> Self {
        self.rasterizer_discard_enable = enable;
        self
    }

    /// Sets the polygon mode.
    pub fn with_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Sets which faces are culled.
    pub fn with_cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    /// Sets the front-face winding.
    pub fn with_front_face(mut self, face: vk::FrontFace) -> Self {
        self.front_face = face;
        self
    }

    /// Enables or disables depth bias.
    pub fn with_depth_bias_enable(mut self, enable: bool) -> Self {
        self.depth_bias_enable = enable;
        self
    }

    /// Sets the constant, clamp, and slope depth-bias factors.
    pub fn with_depth_bias(mut self, constant_factor: f32, clamp: f32, slope_factor: f32) -> Self {
        self.depth_bias_constant_factor = constant_factor;
        self.depth_bias_clamp = clamp;
        self.depth_bias_slope_factor = slope_factor;
        self
    }

    /// Adds rasterization create flags.
    pub fn with_flags(mut self, flags: vk::PipelineRasterizationStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }

    /// Sets the rasterized line width in pixels.
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

    /// Converts this state to Vulkan rasterization state.
    pub fn to_vk_rasterization_state(&self) -> vk::PipelineRasterizationStateCreateInfo<'_> {
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(self.depth_clamp_enable)
            .rasterizer_discard_enable(self.rasterizer_discard_enable)
            .polygon_mode(self.polygon_mode)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(self.depth_bias_enable)
            .depth_bias_constant_factor(self.depth_bias_constant_factor)
            .depth_bias_clamp(self.depth_bias_clamp)
            .depth_bias_slope_factor(self.depth_bias_slope_factor)
            .flags(self.flags)
            .line_width(self.line_width)
    }
}

/// Depth and stencil tests for one graphics pipeline.
pub struct DepthStencilState {
    /// Enables the depth test.
    pub depth_test_enable: bool,
    /// Writes depth when the depth test passes.
    pub depth_write_enable: bool,
    /// Comparison used by the depth test.
    pub depth_compare_op: vk::CompareOp,
    /// Enables the depth-bounds test.
    pub depth_bounds_test_enable: bool,
    /// Enables the stencil test.
    pub stencil_test_enable: bool,
    /// Minimum depth accepted by the depth-bounds test.
    pub min_depth_bounds: f32,
    /// Maximum depth accepted by the depth-bounds test.
    pub max_depth_bounds: f32,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        }
    }
}

impl DepthStencilState {
    /// Enables or disables the depth test.
    pub fn with_depth_test_enable(mut self, enable: bool) -> Self {
        self.depth_test_enable = enable;
        self
    }

    /// Enables or disables depth writes.
    pub fn with_depth_write_enable(mut self, enable: bool) -> Self {
        self.depth_write_enable = enable;
        self
    }

    /// Sets the depth comparison.
    pub fn with_depth_compare_op(mut self, op: vk::CompareOp) -> Self {
        self.depth_compare_op = op;
        self
    }

    /// Enables or disables the depth-bounds test.
    pub fn with_depth_bounds_test_enable(mut self, enable: bool) -> Self {
        self.depth_bounds_test_enable = enable;
        self
    }

    /// Enables or disables the stencil test.
    pub fn with_stencil_test_enable(mut self, enable: bool) -> Self {
        self.stencil_test_enable = enable;
        self
    }

    /// Sets the depth-bounds range.
    pub fn with_depth_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_depth_bounds = min;
        self.max_depth_bounds = max;
        self
    }

    /// Converts this state to Vulkan depth-stencil state.
    pub fn to_vk_depth_stencil_state(&self) -> vk::PipelineDepthStencilStateCreateInfo<'_> {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test_enable)
            .depth_write_enable(self.depth_write_enable)
            .depth_compare_op(self.depth_compare_op)
            .depth_bounds_test_enable(self.depth_bounds_test_enable)
            .stencil_test_enable(self.stencil_test_enable)
            .min_depth_bounds(self.min_depth_bounds)
            .max_depth_bounds(self.max_depth_bounds)
    }
}

/// Multisampling and alpha-to-coverage for one graphics pipeline.
pub struct MultisampleState {
    /// Enables sample shading.
    pub sample_shading_enable: bool,
    /// Samples per pixel.
    pub rasterization_samples: vk::SampleCountFlags,
    /// Fraction of samples shaded when sample shading is enabled.
    pub min_sample_shading: f32,
    /// Sample mask. `None` uses every sample.
    pub sample_mask: Option<Vec<u32>>,
    /// Uses alpha to generate a coverage mask.
    pub alpha_to_coverage_enable: bool,
    /// Replaces fragment alpha with one after coverage is computed.
    pub alpha_to_one_enable: bool,
    /// Multisample create flags.
    pub flags: vk::PipelineMultisampleStateCreateFlags,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            sample_shading_enable: false,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 1.0,
            sample_mask: None,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        }
    }
}

impl MultisampleState {
    /// Enables or disables sample shading.
    pub fn with_sample_shading_enable(mut self, enable: bool) -> Self {
        self.sample_shading_enable = enable;
        self
    }

    /// Sets the sample count.
    pub fn with_rasterization_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.rasterization_samples = samples;
        self
    }

    /// Sets the minimum sample-shading fraction.
    pub fn with_min_sample_shading(mut self, min_shading: f32) -> Self {
        self.min_sample_shading = min_shading;
        self
    }

    /// Sets the sample mask.
    pub fn with_sample_mask(mut self, mask: Vec<u32>) -> Self {
        self.sample_mask = Some(mask);
        self
    }

    /// Enables or disables alpha-to-coverage.
    pub fn with_alpha_to_coverage_enable(mut self, enable: bool) -> Self {
        self.alpha_to_coverage_enable = enable;
        self
    }

    /// Enables or disables alpha-to-one.
    pub fn with_alpha_to_one_enable(mut self, enable: bool) -> Self {
        self.alpha_to_one_enable = enable;
        self
    }

    /// Adds multisample create flags.
    pub fn with_flags(mut self, flags: vk::PipelineMultisampleStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}

/// Blend equation for one color attachment.
pub struct ColorBlendAttachmentState {
    /// Enables blending for this attachment.
    pub blend_enable: bool,
    /// Source factor for the color equation.
    pub src_color_blend_factor: vk::BlendFactor,
    /// Destination factor for the color equation.
    pub dst_color_blend_factor: vk::BlendFactor,
    /// Operator for the color equation.
    pub color_blend_op: vk::BlendOp,
    /// Source factor for the alpha equation.
    pub src_alpha_blend_factor: vk::BlendFactor,
    /// Destination factor for the alpha equation.
    pub dst_alpha_blend_factor: vk::BlendFactor,
    /// Operator for the alpha equation.
    pub alpha_blend_op: vk::BlendOp,
    /// Color components written to this attachment.
    pub color_write_mask: vk::ColorComponentFlags,
}

impl Default for ColorBlendAttachmentState {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        }
    }
}

impl ColorBlendAttachmentState {
    /// Enables or disables blending.
    pub fn with_blend_enable(mut self, enable: bool) -> Self {
        self.blend_enable = enable;
        self
    }

    /// Sets the source color factor.
    pub fn with_src_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_color_blend_factor = factor;
        self
    }

    /// Sets the destination color factor.
    pub fn with_dst_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_color_blend_factor = factor;
        self
    }

    /// Sets the color blend operator.
    pub fn with_color_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.color_blend_op = op;
        self
    }

    /// Sets the source alpha factor.
    pub fn with_src_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_alpha_blend_factor = factor;
        self
    }

    /// Sets the destination alpha factor.
    pub fn with_dst_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_alpha_blend_factor = factor;
        self
    }

    /// Sets the alpha blend operator.
    pub fn with_alpha_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.alpha_blend_op = op;
        self
    }

    /// Sets which color components are written.
    pub fn with_color_write_mask(mut self, mask: vk::ColorComponentFlags) -> Self {
        self.color_write_mask = mask;
        self
    }

    pub(crate) fn to_vk_color_blend_attachment_state(
        &self,
    ) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(self.blend_enable)
            .src_color_blend_factor(self.src_color_blend_factor)
            .dst_color_blend_factor(self.dst_color_blend_factor)
            .color_blend_op(self.color_blend_op)
            .src_alpha_blend_factor(self.src_alpha_blend_factor)
            .dst_alpha_blend_factor(self.dst_alpha_blend_factor)
            .alpha_blend_op(self.alpha_blend_op)
            .color_write_mask(self.color_write_mask)
    }
}

/// Logical operation and per-attachment blend state.
pub struct ColorBlendState {
    /// Enables a logical operation in place of blending.
    pub logic_op_enable: bool,
    /// Logical operation, used when [`Self::logic_op_enable`] is set.
    pub logic_op: vk::LogicOp,
    /// Blend state for each color attachment, in order.
    pub attachments: Vec<ColorBlendAttachmentState>,
    /// Constants available to blend factors that reference them.
    pub blend_constants: [f32; 4],
    /// Color-blend create flags.
    pub flags: vk::PipelineColorBlendStateCreateFlags,
}

impl Default for ColorBlendState {
    fn default() -> Self {
        Self {
            logic_op_enable: false,
            logic_op: vk::LogicOp::CLEAR,
            attachments: Vec::new(),
            blend_constants: [0.0; 4],
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        }
    }
}

impl ColorBlendState {
    /// Enables or disables the logical operation.
    pub fn with_logic_op_enable(mut self, enable: bool) -> Self {
        self.logic_op_enable = enable;
        self
    }

    /// Sets the logical operation.
    pub fn with_logic_op(mut self, op: vk::LogicOp) -> Self {
        self.logic_op = op;
        self
    }

    /// Appends blend state for one color attachment.
    pub fn with_attachment(mut self, attachment: ColorBlendAttachmentState) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Sets the blend constants.
    pub fn with_blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self
    }

    /// Adds color-blend create flags.
    pub fn with_flags(mut self, flags: vk::PipelineColorBlendStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
