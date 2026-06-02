//! Build script: builds the embedded web UI.
//!
//! Order of operations:
//!  1. If `web/dist/index.html` doesn't exist, OR any source under `web/`
//!     is newer than it, run `npm ci` (if needed) and `npm run build`.
//!  2. Re-emit the embedded directory as a cargo build dep so changes
//!     to the frontend trigger a rebuild of the GUI binary.
//!
//! The GUI's `assets.rs` then `#[derive(RustEmbed)]`s `web/dist/` and
//! serves it from the axum router. Running `cargo run -p synapse-gui`
//! without ever running `npm` should still work thanks to this step.

use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=web/package.json");
    println!("cargo:rerun-if-changed=web/package-lock.json");
    println!("cargo:rerun-if-changed=web/vite.config.ts");
    println!("cargo:rerun-if-changed=web/index.html");
    println!("cargo:rerun-if-changed=web/src");
    println!("cargo:rerun-if-changed=web/public");
    println!("cargo:rerun-if-changed=web/dist");

    let dist = Path::new("web/dist");
    let dist_index = dist.join("index.html");

    if needs_rebuild(&dist_index) {
        run_npm_build();
    }

    // Sanity check — the embed fails at compile time without this file,
    // and the failure is harder to read than a clear panic here.
    if !dist_index.exists() {
        panic!(
            "synapse-gui build.rs expected web/dist/index.html after build, but it was missing. \
             Run `cd web && npm run build` manually to see the underlying error."
        );
    }
}

fn needs_rebuild(dist_index: &Path) -> bool {
    if !dist_index.exists() {
        return true;
    }
    let Some(dist_mtime) = file_mtime(dist_index) else {
        return true;
    };

    for candidate in [
        "web/package.json",
        "web/package-lock.json",
        "web/vite.config.ts",
        "web/index.html",
        "web/tsconfig.json",
    ] {
        if let Some(src_mtime) = file_mtime(Path::new(candidate)) {
            if src_mtime > dist_mtime {
                return true;
            }
        }
    }

    if let Ok(entries) = fs::read_dir("web/src") {
        for entry in entries.flatten() {
            if let Some(src_mtime) = file_mtime(&entry.path()) {
                if src_mtime > dist_mtime {
                    return true;
                }
            }
        }
    }

    false
}

fn file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn run_npm_build() {
    eprintln!("[synapse-gui build.rs] building embedded web UI...");

    let npm = which_npm();
    let Some(npm) = npm else {
        panic!(
            "`npm` was not found in PATH. Install Node.js (>= 18) to build the GUI frontend, \
             or pre-build web/dist and re-run."
        );
    };

    if !Path::new("web/node_modules").exists() {
        run_cmd(&npm, &["ci"], "web");
    }
    run_cmd(&npm, &["run", "build"], "web");
}

fn which_npm() -> Option<String> {
    // Try the standard invocations in order; on Windows .cmd wrapping matters.
    for cand in ["npm", "npm.cmd"] {
        let status = Command::new(cand)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Ok(s) = status {
            if s.success() {
                return Some(cand.to_owned());
            }
        }
    }
    None
}

fn run_cmd(prog: &str, args: &[&str], cwd: &str) {
    eprintln!(
        "[synapse-gui build.rs] {prog} {} (cwd={cwd})",
        args.join(" ")
    );
    let status = Command::new(prog)
        .args(args)
        .current_dir(cwd)
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to spawn `{prog}`: {e}");
        });
    if !status.success() {
        panic!(
            "`{prog} {}` exited with {status:?} — see output above.",
            args.join(" ")
        );
    }
}
