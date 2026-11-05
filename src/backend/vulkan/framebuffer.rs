use ash::vk;

use super::device::Device;
use crate::error::vk_error;

/// Framebuffer bound to one render pass and extent.
pub struct Framebuffer {
    pub(crate) framebuffer: vk::Framebuffer,
    pub(crate) extent: vk::Extent2D,
}

impl Framebuffer {
    /// Vulkan framebuffer handle.
    pub fn handle(&self) -> vk::Framebuffer {
        self.framebuffer
    }

    /// Width and height in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Destroys the framebuffer. Attachment views are left in place.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}

/// Collects attachment views and the framebuffer extent.
pub struct FramebufferBuilder {
    attachments: Vec<vk::ImageView>,
    width: u32,
    height: u32,
    layers: u32,
    flags: vk::FramebufferCreateFlags,
}

impl Default for FramebufferBuilder {
    fn default() -> Self {
        Self {
            attachments: Vec::new(),
            width: 0,
            height: 0,
            layers: 1,
            flags: vk::FramebufferCreateFlags::empty(),
        }
    }
}

impl FramebufferBuilder {
    /// Starts a builder with one array layer and no attachments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an attachment view.
    pub fn with_attachment(mut self, attachment: vk::ImageView) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Sets the framebuffer size in pixels.
    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the number of array layers.
    pub fn with_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    /// Adds create flags.
    pub fn with_flags(mut self, flags: vk::FramebufferCreateFlags) -> Self {
        self.flags |= flags;
        self
    }

    /// Creates the framebuffer for `render_pass`.
    pub fn build(
        self,
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<Framebuffer, crate::Error> {
        debug_assert!(
            self.width > 0 && self.height > 0,
            "Rotex Core Panic: Framebuffer dimensions must be greater than zero!"
        );
        if !self.flags.contains(vk::FramebufferCreateFlags::IMAGELESS) {
            debug_assert!(
                !self.attachments.is_empty(),
                "Rotex Core Panic: Standard framebuffers require at least one attachment view!"
            );
        }

        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&self.attachments)
            .width(self.width)
            .height(self.height)
            .layers(self.layers)
            .flags(self.flags);

        let framebuffer = unsafe {
            device
                .logical_device()
                .create_framebuffer(&framebuffer_info, None)
        }
        .map_err(vk_error)?;

        Ok(Framebuffer {
            framebuffer,
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
        })
    }
}
