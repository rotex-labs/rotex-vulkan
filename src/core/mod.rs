use std::ffi::{CStr, CString, c_void};

use ash::vk;

use crate::backend::vulkan::Adapter;
use crate::error::{Error, ErrorKind, vk_error};

#[derive(Debug, Clone)]
pub struct InstanceOptions {
    pub application_name: &'static str,
    pub engine_name: &'static str,
    pub enable_validation: bool,
    pub enable_debug_utils: bool,
}

impl Default for InstanceOptions {
    fn default() -> Self {
        Self {
            application_name: "rotex_vulkan",
            engine_name: "rotex_vulkan",
            enable_validation: false,
            enable_debug_utils: false,
        }
    }
}

pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Instance {
    pub fn new_with_options(
        options: &InstanceOptions,
        extensions: &[*const i8],
    ) -> Result<(Self, Option<DebugMessenger>), Error> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|_| Error::fatal(ErrorKind::Unsupported("Failed to load Vulkan entry")))?;

        let app_name = CString::new(options.application_name)
            .map_err(|_| Error::fatal(ErrorKind::Unsupported("Application name contains NUL")))?;
        let engine_name = CString::new(options.engine_name)
            .map_err(|_| Error::fatal(ErrorKind::Unsupported("Engine name contains NUL")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_0);

        let available_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(vk_error)?;
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.map_err(vk_error)?;

        let mut enabled_extensions = extensions.to_vec();
        if options.enable_debug_utils
            && !enabled_extensions.contains(&ash::ext::debug_utils::NAME.as_ptr())
        {
            if extension_available(&available_extensions, ash::ext::debug_utils::NAME) {
                enabled_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            } else {
                eprintln!("[vulkan] VK_EXT_debug_utils unavailable; skipping debug messenger");
            }
        }

        let mut enabled_layers = Vec::new();
        if options.enable_validation {
            let validation_layer = c"VK_LAYER_KHRONOS_validation";
            if layer_available(&available_layers, validation_layer) {
                enabled_layers.push(validation_layer.as_ptr());
            } else {
                eprintln!("[vulkan] VK_LAYER_KHRONOS_validation unavailable; skipping validation");
            }
        }

        for ext in &enabled_extensions {
            let name = unsafe { CStr::from_ptr(*ext) };
            if !extension_available(&available_extensions, name) {
                eprintln!(
                    "[vulkan] required instance extension {} is not available",
                    name.to_string_lossy()
                );
                if name.to_bytes() == b"VK_KHR_wayland_surface" {
                    eprintln!(
                        "[vulkan] hint: RenderDoc does not support Wayland surfaces; \
                         use X11/XWayland (WINIT_UNIX_BACKEND=x11) when capturing"
                    );
                }
                return Err(Error::fatal(ErrorKind::Unsupported(
                    "Required Vulkan instance extension is not available",
                )));
            }
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extensions)
            .enabled_layer_names(&enabled_layers);

        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(vk_error)?;

        let debug_utils_enabled = enabled_extensions
            .iter()
            .any(|ext| *ext == ash::ext::debug_utils::NAME.as_ptr());
        let debug = if options.enable_debug_utils && debug_utils_enabled {
            Some(DebugMessenger::new(&entry, &instance)?)
        } else {
            None
        };

        Ok((Self { entry, instance }, debug))
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub(crate) fn enumerate_adapters(&self) -> Vec<Adapter> {
        let Ok(physical_devices) = (unsafe { self.instance.enumerate_physical_devices() }) else {
            return Vec::new();
        };

        physical_devices
            .into_iter()
            .map(|handle| {
                let properties = unsafe { self.instance.get_physical_device_properties(handle) };
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                Adapter::new(
                    handle,
                    name,
                    properties.device_type,
                    properties.limits,
                    properties.vendor_id,
                    properties.device_id,
                    properties.driver_version,
                )
            })
            .collect()
    }

    pub fn destroy(self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self, Error> {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        let messenger =
            unsafe { loader.create_debug_utils_messenger(&create_info, None) }.map_err(vk_error)?;
        Ok(Self { loader, messenger })
    }

    pub fn destroy(self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    if !callback_data.is_null() {
        let message_ptr = unsafe { (*callback_data).p_message };
        if !message_ptr.is_null() {
            let message = unsafe { CStr::from_ptr(message_ptr) };
            eprintln!("[vulkan] {}", message.to_string_lossy());
        }
    }
    vk::FALSE
}

fn extension_available(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions.iter().any(|ext| {
        let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
        ext_name.to_bytes() == name.to_bytes()
    })
}

fn layer_available(layers: &[vk::LayerProperties], name: &CStr) -> bool {
    layers.iter().any(|layer| {
        let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        layer_name.to_bytes() == name.to_bytes()
    })
}
