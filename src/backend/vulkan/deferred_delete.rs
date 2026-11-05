use super::buffer::RotexBuffer;
use super::device::Device;
use super::image::RotexImage;

/// GPU resource whose destruction waits for a frame to finish.
pub enum DeferredResource {
    /// Buffer.
    Buffer(RotexBuffer),
    /// Image, including its view and memory.
    Image(RotexImage),
    /// Vertex and index buffers of one mesh.
    Mesh {
        /// Vertex buffer.
        vertex: RotexBuffer,
        /// Index buffer.
        index: RotexBuffer,
    },
}

/// Resources queued until the frame that used them has completed.
#[derive(Default)]
pub struct DeferredDeleteQueue {
    pending: Vec<(usize, DeferredResource)>,
}

impl DeferredDeleteQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Destroys `resource` once `frame_index` has been processed.
    pub fn defer_after_frame(&mut self, frame_index: usize, resource: DeferredResource) {
        self.pending.push((frame_index, resource));
    }

    /// Destroys every resource queued for `completed_frame`.
    pub fn process_frame(&mut self, device: &Device, completed_frame: usize) {
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].0 == completed_frame {
                let (_, resource) = self.pending.swap_remove(index);
                match resource {
                    DeferredResource::Buffer(buffer) => buffer.destroy(device),
                    DeferredResource::Image(image) => image.destroy(device),
                    DeferredResource::Mesh { vertex, index } => {
                        vertex.destroy(device);
                        index.destroy(device);
                    }
                }
            } else {
                index += 1;
            }
        }
    }

    /// Destroys every queued resource immediately.
    pub fn destroy_all(&mut self, device: &Device) {
        for (_, resource) in self.pending.drain(..) {
            match resource {
                DeferredResource::Buffer(buffer) => buffer.destroy(device),
                DeferredResource::Image(image) => image.destroy(device),
                DeferredResource::Mesh { vertex, index } => {
                    vertex.destroy(device);
                    index.destroy(device);
                }
            }
        }
    }
}
