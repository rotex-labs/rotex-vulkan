<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/dark/rotex-vulkan-header.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/light/rotex-vulkan-header.svg">
    <img alt="rotex-vulkan" src="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/light/rotex-vulkan-header.svg" width="1200">
  </picture>
</p>

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

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
