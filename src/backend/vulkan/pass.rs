use ash::vk;

use super::device::Device;

/// Color and depth attachments referenced by one subpass.
#[derive(Debug, Clone)]
pub struct SubpassBlueprint {
    /// Indices into the render-pass attachment list.
    pub color_attachments: Vec<u32>,
    /// Depth attachment index, when the subpass uses one.
    pub depth_attachment: Option<u32>,
}

/// Render pass and the attachment descriptions it was created with.
pub struct RenderPass {
    pub(crate) render_pass: vk::RenderPass,
    attachments: Vec<vk::AttachmentDescription>,
}

impl RenderPass {
    /// Vulkan render-pass handle.
    pub fn handle(&self) -> vk::RenderPass {
        self.render_pass
    }

    /// Attachment descriptions, in creation order.
    pub fn attachments(&self) -> &[vk::AttachmentDescription] {
        &self.attachments
    }

    /// Destroys the render pass.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

/// Attachments, subpasses, and dependencies for one render pass.
#[derive(Default)]
pub struct RenderPassBuilder {
    attachments: Vec<vk::AttachmentDescription>,
    subpasses: Vec<SubpassBlueprint>,
    dependencies: Vec<vk::SubpassDependency>,
}

impl RenderPassBuilder {
    /// Starts an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an attachment.
    pub fn with_attachment(mut self, attachment: vk::AttachmentDescription) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Appends a subpass.
    pub fn with_subpass(mut self, subpass: SubpassBlueprint) -> Self {
        self.subpasses.push(subpass);
        self
    }

    /// Appends a subpass dependency.
    pub fn with_dependency(mut self, dependency: vk::SubpassDependency) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Creates the render pass.
    pub fn build(self, device: &Device) -> Result<RenderPass, vk::Result> {
        let mut all_color_refs: Vec<Vec<vk::AttachmentReference>> =
            Vec::with_capacity(self.subpasses.len());
        let mut all_depth_refs: Vec<Option<vk::AttachmentReference>> =
            Vec::with_capacity(self.subpasses.len());

        for blueprint in &self.subpasses {
            let color_refs: Vec<vk::AttachmentReference> = blueprint
                .color_attachments
                .iter()
                .map(|&idx| {
                    vk::AttachmentReference::default()
                        .attachment(idx)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect();
            all_color_refs.push(color_refs);

            let depth_ref = blueprint.depth_attachment.map(|idx| {
                vk::AttachmentReference::default()
                    .attachment(idx)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            });
            all_depth_refs.push(depth_ref);
        }

        let mut vk_subpasses: Vec<vk::SubpassDescription> =
            Vec::with_capacity(self.subpasses.len());

        for i in 0..self.subpasses.len() {
            let mut subpass_desc = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&all_color_refs[i]);

            if let Some(depth_ref) = &all_depth_refs[i] {
                subpass_desc = subpass_desc.depth_stencil_attachment(depth_ref);
            }

            vk_subpasses.push(subpass_desc);
        }

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&self.attachments)
            .subpasses(&vk_subpasses)
            .dependencies(&self.dependencies);

        let render_pass = unsafe {
            device
                .logical_device()
                .create_render_pass(&render_pass_info, None)?
        };
        Ok(RenderPass {
            render_pass,
            attachments: self.attachments,
        })
    }
}
