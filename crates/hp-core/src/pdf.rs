//! Minimal PDF writer: one image per page plus an optional invisible OCR text layer
//! (making the PDF searchable / selectable). Pure Rust, no external tools.

use std::io::Write;

/// Word placed on a page, in image pixel coordinates (origin top-left).
#[derive(Debug, Clone)]
pub struct TextWord {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

pub enum PageImage {
    /// Baseline JPEG bytes (DCTDecode).
    Jpeg {
        data: Vec<u8>,
        width: u32,
        height: u32,
        gray: bool,
    },
    /// Raw 1-bit image, rows padded to bytes, 1 = white. Compressed with Flate.
    Bilevel {
        packed: Vec<u8>,
        width: u32,
        height: u32,
    },
}

pub struct Page {
    pub image: PageImage,
    pub dpi: f64,
    pub words: Vec<TextWord>,
}

fn escape_pdf_text(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in s.chars() {
        // WinAnsi covers Latin-1; anything else becomes '?'.
        let b = match ch {
            '(' | ')' | '\\' => {
                out.push(b'\\');
                ch as u8
            }
            c if (c as u32) >= 32 && (c as u32) < 127 => c as u8,
            c if (c as u32) >= 160 && (c as u32) <= 255 => c as u32 as u8,
            _ => b'?',
        };
        out.push(b);
    }
    out
}

/// Build a PDF from pages.
pub fn write_pdf(pages: &[Page], title: &str) -> Vec<u8> {
    let mut objects: Vec<Vec<u8>> = Vec::new(); // index i => object i+1
    // Reserve 1: catalog, 2: pages, 3: font, 4: info
    objects.push(Vec::new());
    objects.push(Vec::new());
    objects.push(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    );
    let mut info = b"<< /Producer (Hyper Pixel) /Title (".to_vec();
    info.extend(escape_pdf_text(title));
    info.extend(b") >>");
    objects.push(info);

    let mut page_ids = Vec::new();
    for page in pages {
        let (w_px, h_px) = match &page.image {
            PageImage::Jpeg { width, height, .. } | PageImage::Bilevel { width, height, .. } => {
                (*width, *height)
            }
        };
        let pt_w = w_px as f64 * 72.0 / page.dpi;
        let pt_h = h_px as f64 * 72.0 / page.dpi;

        // Image XObject
        let mut img_obj = Vec::new();
        match &page.image {
            PageImage::Jpeg {
                data,
                width,
                height,
                gray,
            } => {
                write!(
                    img_obj,
                    "<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /{} /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",
                    if *gray { "DeviceGray" } else { "DeviceRGB" },
                    data.len()
                )
                .unwrap();
                img_obj.extend(data);
            }
            PageImage::Bilevel {
                packed,
                width,
                height,
            } => {
                let z = miniz_oxide::deflate::compress_to_vec_zlib(packed, 8);
                write!(
                    img_obj,
                    "<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /FlateDecode /Length {} >>\nstream\n",
                    z.len()
                )
                .unwrap();
                img_obj.extend(z);
            }
        }
        img_obj.extend(b"\nendstream");
        objects.push(img_obj);
        let img_id = objects.len();

        // Content stream
        let mut content = Vec::new();
        write!(content, "q {pt_w:.3} 0 0 {pt_h:.3} 0 0 cm /Im0 Do Q\n").unwrap();
        if !page.words.is_empty() {
            let s = 72.0 / page.dpi;
            content.extend(b"BT 3 Tr\n");
            for word in &page.words {
                if word.text.trim().is_empty() {
                    continue;
                }
                let size = (word.h as f64 * s * 0.9).max(1.0);
                // Helvetica averages ~0.5em per char; scale horizontally to fit the box.
                let natural = word.text.chars().count() as f64 * size * 0.5;
                let tz = if natural > 0.0 {
                    (word.w as f64 * s / natural * 100.0).clamp(10.0, 500.0)
                } else {
                    100.0
                };
                let x = word.x as f64 * s;
                let y = pt_h - (word.y as f64 + word.h as f64 * 0.85) * s;
                write!(
                    content,
                    "/F1 {size:.2} Tf {tz:.1} Tz 1 0 0 1 {x:.2} {y:.2} Tm ("
                )
                .unwrap();
                content.extend(escape_pdf_text(&word.text));
                content.extend(b") Tj\n");
            }
            content.extend(b"ET\n");
        }
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&content, 6);
        let mut cobj =
            format!("<< /Length {} /Filter /FlateDecode >>\nstream\n", z.len()).into_bytes();
        cobj.extend(z);
        cobj.extend(b"\nendstream");
        objects.push(cobj);
        let content_id = objects.len();

        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pt_w:.3} {pt_h:.3}] /Resources << /XObject << /Im0 {img_id} 0 R >> /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            )
            .into_bytes(),
        );
        page_ids.push(objects.len());
    }
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    let kids: Vec<String> = page_ids.iter().map(|i| format!("{i} 0 R")).collect();
    objects[1] = format!(
        "<< /Type /Pages /Kids [{}] /Count {} >>",
        kids.join(" "),
        page_ids.len()
    )
    .into_bytes();

    let mut out = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        offsets.push(out.len());
        write!(out, "{} 0 obj\n", i + 1).unwrap();
        out.extend(obj);
        out.extend(b"\nendobj\n");
    }
    let xref = out.len();
    write!(out, "xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).unwrap();
    for o in offsets {
        write!(out, "{o:010} 00000 n \n").unwrap();
    }
    write!(
        out,
        "trailer\n<< /Size {} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    )
    .unwrap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_valid_looking_pdf() {
        let packed = vec![0xFFu8; 2 * 16];
        let pdf = write_pdf(
            &[Page {
                image: PageImage::Bilevel {
                    packed,
                    width: 16,
                    height: 16,
                },
                dpi: 72.0,
                words: vec![TextWord {
                    text: "Hello (world)".into(),
                    x: 1.0,
                    y: 1.0,
                    w: 10.0,
                    h: 4.0,
                }],
            }],
            "Test",
        );
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.starts_with("%PDF-1.4"));
        assert!(s.contains("/Count 1"));
        assert!(s.trim_end().ends_with("%%EOF"));
    }
}
