mod support;

use support::env::with_env_vars;

#[test]
fn scoped_env_restores_exact_value_after_panic_and_poison() {
    const KEY: &str = "VORONOI_MESH_TEST_ENV_RESTORE";

    let process_value = std::env::var_os(KEY);
    std::env::set_var(KEY, "before");

    let panic = std::panic::catch_unwind(|| {
        with_env_vars(&[(KEY, Some("during"))], || {
            assert_eq!(
                std::env::var_os(KEY).as_deref(),
                Some(std::ffi::OsStr::new("during"))
            );
            panic!("exercise unwind restoration");
        });
    });
    let after_panic = std::env::var_os(KEY);

    let absent_inside = with_env_vars(&[(KEY, None)], || std::env::var_os(KEY).is_none());
    let after_poison_recovery = std::env::var_os(KEY);

    match process_value {
        Some(value) => std::env::set_var(KEY, value),
        None => std::env::remove_var(KEY),
    }

    assert!(panic.is_err());
    assert_eq!(after_panic.as_deref(), Some(std::ffi::OsStr::new("before")));
    assert!(absent_inside);
    assert_eq!(
        after_poison_recovery.as_deref(),
        Some(std::ffi::OsStr::new("before"))
    );
}
