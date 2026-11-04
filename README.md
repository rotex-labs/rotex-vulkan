# rotex-vulkan

Stateless Vulkan translator for the same commands as the other backends.

> [!WARNING]
> **Beta, pre-1.0.** Rotex is under active development. APIs, behavior, and crate
> boundaries can change in any 0.x release. Do not depend on it for production work yet.

## Overview

`rotex-vulkan` implements `GpuPlatform` and `GpuBackend` with Vulkan. It translates the `RhiCommand` list the user recorded. It does not read a scene or insert barriers the user did not emit.

## Responsibilities

- `VulkanGpuPlatform` and `VulkanBridge`
- Handle maps, a pipeline cache, and frame-slot submission
- Native barriers, copies, queries, storage buffers, and indexed indirect draw

## Dependencies

`rotex-core` and `rotex-types`. Tests use `rotex-conformance`.

## Status

Version 0.2.0. A texture render target can be cleared without a swapchain. Presentation still requires an attached surface.

## Part of Rotex

Rotex is a graphics HAL workspace. See [rotex-dev](https://github.com/rotex-labs/rotex-dev) for the architecture and the other crates.

## License

Licensed under [MIT](LICENSE).
