//! Chat-template prompt formatting.
//!
//! A chat template is model-specific. Tiny random fixtures use plain text;
//! SmolLM2 Instruct uses ChatML. Unknown templates require explicit raw mode.

/// Format only model families with a known template. This does not certify
/// weight/backend compatibility, which the loader checks independently.
pub fn format_prompt_for_model(
    model_id: &str,
    user_input: &str,
    raw: bool,
) -> anyhow::Result<String> {
    if raw
        || model_id == "hf-internal-testing/tiny-random-LlamaForCausalLM"
        || model_id == "synapse/tiny-llama-fixture"
    {
        return Ok(user_input.to_owned());
    }
    if model_id.starts_with("HuggingFaceTB/SmolLM2-") && model_id.ends_with("-Instruct") {
        return Ok(format_instruct_prompt(user_input));
    }
    anyhow::bail!("no validated chat template for {model_id}; select raw prompt mode")
}

/// Wrap a user message in the SmolLM2 Instruct chat template.
///
/// Returns a string that can be fed directly to
/// [`crate::inference::SimpleTokenizer::encode`].
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

    #[test]
    fn template_selection_does_not_apply_chatml_to_arbitrary_models() {
        assert_eq!(
            format_prompt_for_model("synapse/tiny-llama-fixture", "Hello", false).unwrap(),
            "Hello"
        );
        assert!(format_prompt_for_model("unreviewed/model", "Hello", false).is_err());
        assert_eq!(
            format_prompt_for_model("unreviewed/model", "Hello", true).unwrap(),
            "Hello"
        );
        assert!(
            format_prompt_for_model("HuggingFaceTB/SmolLM2-135M-Instruct", "Hello", false)
                .unwrap()
                .starts_with("<|im_start|>")
        );
    }
}
