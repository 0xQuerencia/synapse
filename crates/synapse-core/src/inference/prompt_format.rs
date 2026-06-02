//! Chat-template prompt formatting.
//!
//! Synapse is currently validated against
//! `hf-internal-testing/tiny-random-LlamaForCausalLM`, whose
//! tokenizer expects the SmolLM2 chat template. Centralising the
//! template here keeps the CLI and any future GUI from drifting.

/// Wrap a user message in the model's chat template.
///
/// Returns a string that can be fed directly to
/// [`SimpleTokenizer::encode`].
pub fn format_instruct_prompt(user_input: &str) -> String {
    format!("<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n")
}

/// Interpret C-style escape sequences (`\n`, `\t`) in raw prompt
/// strings, as used by the CLI `--prompt` flag.
pub fn unescape_prompt(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_user_in_smol_template() {
        let formatted = format_instruct_prompt("Hello");
        assert_eq!(
            formatted,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn unescapes_newlines_and_tabs() {
        let raw = "line1\\nline2\\tcol";
        assert_eq!(unescape_prompt(raw), "line1\nline2\tcol");
    }
}
