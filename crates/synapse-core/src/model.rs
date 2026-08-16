use crate::protocol::ProtocolError;
use serde::{Deserialize, Serialize};

pub const MAX_MODEL_ID_LEN: usize = 256;
const MAX_REVISION_LEN: usize = 128;
const SHA256_HEX_LEN: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub id: String,
    pub revision: Option<String>,
    pub sha256: Option<String>,
}

impl ModelIdentity {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            revision: None,
            sha256: None,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        validate_printable("model id", &self.id, 1, MAX_MODEL_ID_LEN)?;

        if let Some(revision) = &self.revision {
            validate_printable("model revision", revision, 1, MAX_REVISION_LEN)?;
        }

        if let Some(sha256) = &self.sha256
            && (sha256.len() != SHA256_HEX_LEN || !sha256.chars().all(|c| c.is_ascii_hexdigit()))
        {
            return Err(ProtocolError::InvalidField {
                field: "model sha256",
                reason: "expected 64 lowercase or uppercase hex characters",
            });
        }

        Ok(())
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    crate::hash::sha256_hex(data)
}

pub(crate) fn validate_printable(
    field: &'static str,
    value: &str,
    min_len: usize,
    max_len: usize,
) -> Result<(), ProtocolError> {
    if value.len() < min_len || value.len() > max_len {
        return Err(ProtocolError::InvalidField {
            field,
            reason: "length outside accepted range",
        });
    }

    if value.chars().any(|c| c.is_control()) {
        return Err(ProtocolError::InvalidField {
            field,
            reason: "control characters are not allowed",
        });
    }

    Ok(())
}
