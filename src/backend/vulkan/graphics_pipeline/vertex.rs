use ash::vk;

/// Vertex bindings and attributes for one pipeline.
pub struct VertexInputDescriptor {
    /// Binding slots and their strides.
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    /// Shader locations sourced from those bindings.
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    /// Vertex-input create flags.
    pub flags: vk::PipelineVertexInputStateCreateFlags,
}

/// Vertex type that can describe its pipeline input layout.
pub trait Vertex {
    /// Bindings and attributes for this vertex type.
    fn descriptor() -> VertexInputDescriptor;
}

impl Default for VertexInputDescriptor {
    fn default() -> Self {
        Self {
            binding_descriptions: Vec::new(),
            attribute_descriptions: Vec::new(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        }
    }
}

impl VertexInputDescriptor {
    /// Appends a binding.
    pub fn with_binding(mut self, description: vk::VertexInputBindingDescription) -> Self {
        self.binding_descriptions.push(description);
        self
    }

    /// Appends an attribute.
    pub fn with_attribute(mut self, description: vk::VertexInputAttributeDescription) -> Self {
        self.attribute_descriptions.push(description);
        self
    }

    /// Adds vertex-input create flags.
    pub fn with_flags(mut self, flags: vk::PipelineVertexInputStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
