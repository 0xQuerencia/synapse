//! Build script: verifies the embedded HTML asset is present.
//!
//! As of v3 the GUI is a single self-contained `web/index.html`
//! (Tailwind via CDN, vanilla JS, no build step). The earlier
//! React/Vite build pipeline was retired; this script is now a
//! guard that fails loudly if the file is missing rather than
//! running an external package manager.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=web/index.html");

    let index = Path::new("web/index.html");
    if !index.exists() {
        panic!(
            "synapse-gui build.rs expected web/index.html but it was missing. \
             The current GUI is a self-contained HTML file; restore it under \
             crates/synapse-gui/web/index.html."
        );
    }
}
