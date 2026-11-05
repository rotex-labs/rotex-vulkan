use std::fmt::{Display, Formatter};

use ash::vk;

/// How serious a backend error is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Diagnostic that does not change control flow.
    Info,
    /// Recoverable condition worth reporting.
    Warning,
    /// The call failed and the device can still be used.
    Recoverable,
    /// The call failed and the device should be torn down.
    Fatal,
}

/// Why a Vulkan backend call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Vulkan returned this result.
    Vulkan(vk::Result),
    /// The operation is not available. The string says what was missing.
    Unsupported(&'static str),
    /// No physical device matched the request.
    NoCompatibleDevice,
}

/// Error returned by the Vulkan backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Why the call failed.
    pub kind: ErrorKind,
    /// Whether the device can keep running.
    pub severity: Severity,
}

impl Error {
    /// Builds an error with [`Severity::Fatal`].
    pub fn fatal(kind: ErrorKind) -> Self {
        Self {
            kind,
            severity: Severity::Fatal,
        }
    }

    /// Fatal error for an operation this backend cannot perform.
    pub fn unsupported(message: &'static str) -> Self {
        Self::fatal(ErrorKind::Unsupported(message))
    }

    /// Raw Vulkan result code when [`Self::kind`] is [`ErrorKind::Vulkan`].
    pub fn vk_result_code(&self) -> Option<i32> {
        match self.kind {
            ErrorKind::Vulkan(code) => Some(code.as_raw()),
            _ => None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ErrorKind::Vulkan(code) => write!(f, "Vulkan error: {code:?} ({})", code.as_raw()),
            ErrorKind::Unsupported(message) => write!(f, "Unsupported: {message}"),
            ErrorKind::NoCompatibleDevice => write!(f, "No compatible Vulkan device found"),
        }
    }
}

impl std::error::Error for Error {}

/// Fatal error wrapping a Vulkan result.
pub fn vk_error(result: vk::Result) -> Error {
    Error::fatal(ErrorKind::Vulkan(result))
}
