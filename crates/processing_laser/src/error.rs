//! Thread-local error reporting, mirroring processing_ffi's convention:
//! every FFI call clears the slot, runs under catch_unwind, and the caller
//! checks `laser_check_error()` afterwards.

use std::{
    cell::RefCell,
    ffi::{CString, c_char},
    panic,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Check if the last operation on this thread resulted in an error. Returns a
/// pointer to an error message, or null if there was no error. The pointer is
/// valid until the next processing_laser call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn laser_check_error() -> *const c_char {
    LAST_ERROR.with(|last| {
        last.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

pub fn set_error(error_msg: &str) {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = Some(CString::new(error_msg).unwrap_or_else(|_| {
            CString::new("Failed to allocate error message".to_string()).unwrap()
        }));
    });
}

pub fn clear_error() {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = None;
    });
}

/// Run an FFI body, setting the thread-local error on failure or panic.
pub fn check<T, F>(f: F) -> Option<T>
where
    F: FnOnce() -> Result<T, String>,
{
    // catch panics to prevent unwinding across the FFI boundary
    panic::catch_unwind(panic::AssertUnwindSafe(|| match f() {
        Ok(value) => Some(value),
        Err(err) => {
            set_error(&err);
            None
        }
    }))
    .unwrap_or_else(|e| {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&'static str>() {
            s.to_string()
        } else {
            "Unknown panic payload".to_string()
        };
        set_error(&format!("Panic occurred: {}", msg));
        None
    })
}
