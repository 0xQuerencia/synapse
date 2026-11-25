//! Bound GGUF header declarations before Candle allocates their strings/arrays,
//! then verify tensor storage extents before any weight is materialized.
use anyhow::{Result, ensure};
use candle_core::quantized::gguf_file;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Seek, SeekFrom},
};
const MAX_HEADER: u64 = 64 * 1024 * 1024;
struct Scan<'a> {
    file: &'a mut File,
    length: u64,
    values: u64,
}
impl Scan<'_> {
    fn u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.file.read_exact(&mut bytes)?;
        self.bound()?;
        Ok(u32::from_le_bytes(bytes))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        self.file.read_exact(&mut bytes)?;
        self.bound()?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn bound(&mut self) -> Result<()> {
        ensure!(
            self.file.stream_position()? <= MAX_HEADER,
            "GGUF header exceeds limit"
        );
        Ok(())
    }
    fn skip(&mut self, bytes: u64) -> Result<()> {
        let end = self
            .file
            .stream_position()?
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("GGUF length overflow"))?;
        ensure!(
            end <= self.length && end <= MAX_HEADER,
            "GGUF header length outside file or bound"
        );
        self.file.seek(SeekFrom::Start(end))?;
        Ok(())
    }
    fn string(&mut self, small: bool) -> Result<Option<String>> {
        let len = self.u64()?;
        ensure!(
            len <= if small { 512 } else { 1024 * 1024 },
            "GGUF string length exceeds limit"
        );
        if !small {
            self.skip(len)?;
            return Ok(None);
        }
        let mut bytes = vec![0; len as usize];
        self.file.read_exact(&mut bytes)?;
        self.bound()?;
        let value = String::from_utf8(bytes)?;
        ensure!(
            !value.contains('\0') && !value.is_empty(),
            "invalid GGUF key"
        );
        Ok(Some(value))
    }
    fn value(&mut self, kind: u32, depth: u32) -> Result<()> {
        self.values += 1;
        ensure!(
            self.values <= 2_000_000 && depth <= 8,
            "GGUF metadata complexity exceeds limit"
        );
        match kind {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4..=6 => self.skip(4),
            10..=12 => self.skip(8),
            8 => {
                self.string(false)?;
                Ok(())
            }
            9 => {
                let item = self.u32()?;
                let count = self.u64()?;
                ensure!(count <= 1_000_000, "GGUF array length exceeds limit");
                for _ in 0..count {
                    self.value(item, depth + 1)?;
                }
                Ok(())
            }
            _ => anyhow::bail!("unsupported GGUF metadata type"),
        }
    }
}
pub fn read_checked(file: &mut File) -> Result<gguf_file::Content> {
    file.seek(SeekFrom::Start(0))?;
    let length = file.metadata()?.len();
    let mut magic = [0; 4];
    file.read_exact(&mut magic)?;
    ensure!(&magic == b"GGUF", "invalid GGUF magic");
    let mut scan = Scan {
        file,
        length,
        values: 0,
    };
    let version = scan.u32()?;
    ensure!(
        version == 2 || version == 3,
        "only bounded GGUF v2/v3 supported"
    );
    let tensors = scan.u64()?;
    let metadata = scan.u64()?;
    ensure!(
        (1..=16_384).contains(&tensors) && metadata <= 65_536,
        "GGUF entry count exceeds limit"
    );
    let mut keys = BTreeSet::new();
    for _ in 0..metadata {
        let key = scan.string(true)?.expect("small string");
        ensure!(keys.insert(key.clone()), "duplicate GGUF metadata key");
        let kind = scan.u32()?;
        if key == "general.alignment" {
            let alignment = match kind {
                4 => scan.u32()? as u64,
                10 => scan.u64()?,
                _ => anyhow::bail!("invalid GGUF alignment type"),
            };
            ensure!(
                alignment.is_power_of_two() && alignment <= 4096,
                "invalid GGUF alignment"
            );
        } else {
            scan.value(kind, 0)?;
        }
    }
    keys.clear();
    for _ in 0..tensors {
        let key = scan.string(true)?.expect("small string");
        ensure!(keys.insert(key), "duplicate GGUF tensor name");
        let dims = scan.u32()?;
        ensure!((1..=4).contains(&dims), "invalid GGUF tensor rank");
        let mut elements = 1u64;
        for _ in 0..dims {
            let dim = scan.u64()?;
            ensure!(dim > 0 && dim <= 1_000_000, "invalid GGUF tensor dimension");
            elements = elements
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("GGUF shape overflow"))?;
        }
        ensure!(elements <= 1u64 << 36, "GGUF tensor too large");
        let _kind = scan.u32()?;
        let offset = scan.u64()?;
        ensure!(offset < length, "GGUF tensor offset outside file");
    }
    scan.file.seek(SeekFrom::Start(0))?;
    let content = gguf_file::Content::read(scan.file)?;
    ensure!(
        content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .is_some_and(|v| v == "llama"),
        "unsupported or missing GGUF architecture"
    );
    let mut extents = Vec::with_capacity(content.tensor_infos.len());
    for tensor in content.tensor_infos.values() {
        let elements = tensor.shape.elem_count() as u64;
        let block = tensor.ggml_dtype.block_size() as u64;
        ensure!(
            elements.is_multiple_of(block),
            "GGUF tensor has incomplete quantization block"
        );
        let bytes = (elements / block)
            .checked_mul(tensor.ggml_dtype.type_size() as u64)
            .ok_or_else(|| anyhow::anyhow!("GGUF storage overflow"))?;
        let start = content
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| anyhow::anyhow!("GGUF offset overflow"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("GGUF extent overflow"))?;
        ensure!(end <= length, "truncated GGUF tensor storage");
        extents.push((start, end));
    }
    extents.sort_unstable();
    ensure!(
        extents.windows(2).all(|w| w[0].1 <= w[1].0),
        "overlapping GGUF tensor storage"
    );
    Ok(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rejected(bytes: &[u8]) -> bool {
        let mut file = tempfile::tempfile().unwrap();
        std::io::Write::write_all(&mut file, bytes).unwrap();
        read_checked(&mut file).is_err()
    }
    #[test]
    fn hostile_header_sizes_fail_before_upstream_allocation() {
        let mut prefix = b"GGUF".to_vec();
        prefix.extend(3u32.to_le_bytes());
        prefix.extend(1u64.to_le_bytes());
        prefix.extend(1u64.to_le_bytes());
        let mut key = prefix.clone();
        key.extend(u64::MAX.to_le_bytes());
        assert!(rejected(&key));
        let mut array = prefix;
        array.extend(1u64.to_le_bytes());
        array.extend(b"k");
        array.extend(9u32.to_le_bytes());
        array.extend(8u32.to_le_bytes());
        array.extend(u64::MAX.to_le_bytes());
        assert!(rejected(&array));
        let mut counts = b"GGUF".to_vec();
        counts.extend(3u32.to_le_bytes());
        counts.extend(u64::MAX.to_le_bytes());
        counts.extend(0u64.to_le_bytes());
        assert!(rejected(&counts));
    }
}
