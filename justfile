# Synapse development recipes
# Install just: https://github.com/casey/just

# Default: build the workspace
default: build

# Build the workspace
build:
    cargo build

# Run all tests
test:
    cargo test

# Format code
fmt:
    cargo fmt --all

# Run clippy lints
clippy:
    cargo clippy --workspace -- -D warnings

# Run the localhost demo (single-process validation)
demo *ARGS='--model hf-internal-testing/tiny-random-LlamaForCausalLM --prompt "Hello"':
    cargo run -- demo {{ARGS}}

# Start a master host node
host *ARGS='--master --model HuggingFaceTB/SmolLM2-135M-Instruct --prompt "Hello"':
    cargo run -- host {{ARGS}}

# Start a worker node (requires master address and pairing code)
join *ARGS:
    cargo run -- join {{ARGS}}

# Full CI pipeline: fmt + clippy + test
ci: fmt clippy test

# Clean build artifacts
clean:
    cargo clean

# Generate documentation
docs:
    cargo doc --workspace --open

# Install the CLI locally
install:
    cargo install --path crates/synapse-cli
