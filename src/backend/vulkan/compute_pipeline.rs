use ash::vk;

use super::device::Device;
use crate::Error;
use crate::error::vk_error;

/// Compiled compute pipeline.
pub struct ComputePipeline {
    handle: vk::Pipeline,
}

impl ComputePipeline {
    /// Creates a compute pipeline from `stage` and `layout`.
    pub fn new(
        device: &Device,
        stage: &vk::PipelineShaderStageCreateInfo<'_>,
        layout: vk::PipelineLayout,
    ) -> Result<Self, Error> {
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(*stage)
            .layout(layout);

        let pipelines = unsafe {
            device.logical_device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        }
        .map_err(|(_, err)| vk_error(err))?;

        Ok(Self {
            handle: pipelines[0],
        })
    }

    /// Vulkan pipeline handle.
    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    /// Destroys the pipeline.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_pipeline(self.handle, None);
        }
    }
}
