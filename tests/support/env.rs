//! Process-environment isolation for integration tests.
//!
//! Each integration-test target is a separate process, so this lock only
//! needs to serialize mutations within one target. The scoped restore is
//! still essential: tests in that target share an environment and may run in
//! parallel, while callers may already have supplied the same knob.

// `tests/support` is compiled into every integration-test executable, while
// only targets that mutate process state use this module.
#![allow(dead_code)]

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvRestore {
    saved: Vec<(OsString, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Run `f` with process-environment overrides, restoring the exact prior
/// values during both ordinary return and panic unwinding.
///
/// `Some(value)` sets a variable and `None` removes it for the duration of
/// the closure. Tests in the same integration target that read these knobs
/// must also use this helper, with an empty override list when necessary, so
/// their computation participates in the same serialization boundary.
pub(crate) fn with_env_vars<R>(overrides: &[(&str, Option<&str>)], f: impl FnOnce() -> R) -> R {
    let lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let saved = overrides
        .iter()
        .map(|(key, _)| (OsString::from(key), std::env::var_os(key)))
        .collect();
    let _restore = EnvRestore { saved, _lock: lock };

    for (key, value) in overrides {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    f()
}
