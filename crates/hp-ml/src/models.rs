//! Model files: where they are looked up, how they are downloaded and verified.
//!
//! Models are looked up in, in order:
//! 1. `$HYPER_PIXEL_MODELS`
//! 2. `$APPDIR/usr/share/hyper-pixel/models` (bundled inside the AppImage)
//! 3. `<exe dir>/../share/hyper-pixel/models`
//! 4. `$XDG_DATA_HOME/hyper-pixel/models` (downloaded on demand)

use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("{0} model not found; download it first")]
    Missing(&'static str),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("downloaded file {0} is corrupt (checksum mismatch)")]
    Checksum(String),
    #[error("model failed: {0}")]
    Engine(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ModelError>;

/// One file of a model, with its SHA-256 and size.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    pub name: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum Source {
    /// Each file is downloaded from `<base>/<file name>`.
    Direct { base: &'static str },
    /// A zip archive holding the files under `prefix`.
    Zip {
        url: &'static str,
        prefix: &'static str,
        size: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub label: &'static str,
    pub files: &'static [ModelFile],
    pub source: Source,
    pub licence: &'static str,
}

impl ModelSpec {
    /// Total download size in bytes.
    pub fn download_size(&self) -> u64 {
        match self.source {
            Source::Direct { .. } => self.files.iter().map(|f| f.size).sum(),
            Source::Zip { size, .. } => size,
        }
    }
    fn present_in(&self, dir: &Path) -> bool {
        self.files.iter().all(|f| {
            std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.is_file() && m.len() == f.size)
        })
    }
    /// Directory holding all of this model's files, if any.
    pub fn find(&self) -> Option<PathBuf> {
        model_dirs().into_iter().find(|d| self.present_in(d))
    }
}

/// Big LaMa inpainting (Apache-2.0), exported to ONNX by Carve, fixed 512 x 512 input.
pub const LAMA: ModelSpec = ModelSpec {
    label: "LaMa inpainting",
    files: &[ModelFile {
        name: "lama_fp32.onnx",
        sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        size: 208_044_816,
    }],
    source: Source::Direct {
        base: "https://huggingface.co/Carve/LaMa-ONNX/resolve/main",
    },
    licence: "Apache-2.0 (advimman/lama; ONNX export by Carve)",
};

/// DDColor (paper tiny, Apache-2.0) as exported by Qualcomm AI Hub, 256 x 256 input.
pub const DDCOLOR: ModelSpec = ModelSpec {
    label: "DDColor colourisation",
    files: &[
        ModelFile {
            name: "ddcolor.onnx",
            sha256: "310d0c66a38cd8cfd7d9d93fdb5500fef79d584fe59e25a417aea5915a6b3733",
            size: 2_750_675,
        },
        ModelFile {
            name: "ddcolor.data",
            sha256: "5b65cb3eab9fa4ae9ecacd539b148fa51157655732022913b1cf47346f93940c",
            size: 225_312_128,
        },
    ],
    source: Source::Zip {
        url: "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/ddcolor/releases/v0.66.0/ddcolor-onnx-float.zip",
        prefix: "ddcolor-onnx-float/",
        size: 206_251_342,
    },
    licence: "Apache-2.0 (piddnad/DDColor; ONNX export by Qualcomm AI Hub)",
};

/// Where downloaded models are stored.
pub fn user_model_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hyper-pixel")
        .join("models")
}

/// Candidate model directories, in lookup order.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("HYPER_PIXEL_MODELS") {
        cands.push(p.into());
    }
    if let Some(p) = std::env::var_os("APPDIR") {
        cands.push(PathBuf::from(p).join("usr/share/hyper-pixel/models"));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin) = exe.parent()
    {
        cands.push(bin.join("../share/hyper-pixel/models"));
    }
    cands.push(user_model_dir());
    cands
}

/// Download `url` to `dest` (via a `.part` file). `progress` receives (bytes done, total).
pub fn download_file(url: &str, dest: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    log::info!("downloading {url}");
    let mut resp = ureq::get(url)
        .call()
        .map_err(|e| ModelError::Download(e.to_string()))?;
    let total = resp.body().content_length().unwrap_or(0);
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn verify(dir: &Path, f: &ModelFile) -> Result<()> {
    let p = dir.join(f.name);
    if sha256_file(&p)? != f.sha256 {
        let _ = std::fs::remove_file(&p);
        return Err(ModelError::Checksum(f.name.into()));
    }
    Ok(())
}

/// Download a model into [`user_model_dir`] unless it is already available.
/// `progress` receives (bytes done, total) for the whole model.
pub fn download(spec: &ModelSpec, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    if let Some(d) = spec.find() {
        return Ok(d);
    }
    let dir = user_model_dir();
    std::fs::create_dir_all(&dir)?;
    let total = spec.download_size();
    match spec.source {
        Source::Direct { base } => {
            let mut offset = 0u64;
            for f in spec.files {
                let dest = dir.join(f.name);
                let ok = std::fs::metadata(&dest).is_ok_and(|m| m.len() == f.size);
                if !ok {
                    download_file(&format!("{base}/{}", f.name), &dest, &mut |d, _| {
                        progress(offset + d, total)
                    })?;
                }
                verify(&dir, f)?;
                offset += f.size;
            }
        }
        Source::Zip { url, prefix, .. } => {
            let zip = dir.join(format!("{}.zip", spec.files[0].name));
            download_file(url, &zip, &mut |d, _| progress(d, total))?;
            let res = (|| -> Result<()> {
                for f in spec.files {
                    extract_zip_entry(&zip, &format!("{prefix}{}", f.name), &dir.join(f.name))?;
                    verify(&dir, f)?;
                }
                Ok(())
            })();
            let _ = std::fs::remove_file(&zip);
            res?;
        }
    }
    Ok(dir)
}

/// Extract one stored or deflated entry of a (non-zip64) zip archive, streaming.
fn extract_zip_entry(zip: &Path, name: &str, dest: &Path) -> Result<()> {
    let bad = |m: &str| ModelError::Download(format!("bad zip archive: {m}"));
    let mut f = std::fs::File::open(zip)?;
    let len = f.metadata()?.len();
    // End of central directory: last 22..(22 + 65535) bytes.
    let tail_len = len.min(22 + 65_535);
    f.seek(SeekFrom::Start(len - tail_len))?;
    let mut tail = vec![0u8; tail_len as usize];
    f.read_exact(&mut tail)?;
    let eocd = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&i| tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| bad("no end record"))?;
    let u16le = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as u64;
    let u32le =
        |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as u64;
    let cd_size = u32le(&tail, eocd + 12);
    let cd_off = u32le(&tail, eocd + 16);
    f.seek(SeekFrom::Start(cd_off))?;
    let mut cd = vec![0u8; cd_size as usize];
    f.read_exact(&mut cd)?;
    let mut p = 0usize;
    let mut entry = None;
    while p + 46 <= cd.len() && cd[p..p + 4] == [0x50, 0x4b, 0x01, 0x02] {
        let method = u16le(&cd, p + 10);
        let csize = u32le(&cd, p + 20);
        let usize_ = u32le(&cd, p + 24);
        let nlen = u16le(&cd, p + 28) as usize;
        let elen = u16le(&cd, p + 30) as usize;
        let clen = u16le(&cd, p + 32) as usize;
        let off = u32le(&cd, p + 42);
        if &cd[p + 46..p + 46 + nlen] == name.as_bytes() {
            entry = Some((method, csize, usize_, off));
            break;
        }
        p += 46 + nlen + elen + clen;
    }
    let (method, csize, usize_, off) = entry.ok_or_else(|| bad("entry missing"))?;
    let mut lh = [0u8; 30];
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut lh)?;
    let data = off + 30 + u16le(&lh, 26) + u16le(&lh, 28);
    f.seek(SeekFrom::Start(data))?;
    let mut input = f.take(csize);
    let tmp = dest.with_extension("part");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    match method {
        0 => {
            std::io::copy(&mut input, &mut out)?;
        }
        8 => {
            use miniz_oxide::inflate::stream::{InflateState, inflate};
            use miniz_oxide::{DataFormat, MZFlush, MZStatus};
            let mut st = InflateState::new_boxed(DataFormat::Raw);
            let mut inbuf = vec![0u8; 256 * 1024];
            let mut outbuf = vec![0u8; 256 * 1024];
            let (mut have, mut start) = (0usize, 0usize);
            let mut written = 0u64;
            loop {
                if start == have {
                    have = input.read(&mut inbuf)?;
                    start = 0;
                }
                let flush = if have == 0 {
                    MZFlush::Finish
                } else {
                    MZFlush::None
                };
                let r = inflate(&mut st, &inbuf[start..have], &mut outbuf, flush);
                start += r.bytes_consumed;
                out.write_all(&outbuf[..r.bytes_written])?;
                written += r.bytes_written as u64;
                match r.status {
                    Ok(MZStatus::StreamEnd) => break,
                    Ok(_) => {}
                    Err(_) if have == 0 && r.bytes_written == 0 => {
                        return Err(bad("truncated data"));
                    }
                    Err(miniz_oxide::MZError::Buf) => {}
                    Err(e) => return Err(bad(&format!("inflate {e:?}"))),
                }
                if have == 0 && r.bytes_written == 0 && r.bytes_consumed == 0 {
                    break;
                }
            }
            if written != usize_ {
                return Err(bad("size mismatch"));
            }
        }
        _ => return Err(bad("unsupported compression")),
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deflated zip, built with Python's zipfile: `a/x.txt` = "hello hello hello\n".
    #[test]
    fn extracts_deflated_entry() {
        let dir = std::env::temp_dir().join(format!("hp-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let zip = dir.join("t.zip");
        std::fs::write(&zip, TEST_ZIP).unwrap();
        let out = dir.join("x.txt");
        extract_zip_entry(&zip, "a/x.txt", &out).unwrap();
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            "hello hello hello\n"
        );
        assert!(extract_zip_entry(&zip, "missing", &out).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const TEST_ZIP: &[u8] = include_bytes!("../tests/data/test.zip");
}
