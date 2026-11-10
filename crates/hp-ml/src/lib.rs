//! Neural models for photo restoration, run on the CPU with the pure-Rust `rten` engine:
//! LaMa inpainting (large dust specks and scratches) and DDColor colourisation (black and
//! white photos). Models are downloaded on demand (see [`models`]).

pub mod colorize;
pub mod inpaint;
pub mod models;

pub use colorize::Colorizer;
pub use inpaint::{InpaintParams, Inpainter};
pub use models::{DDCOLOR, LAMA, ModelError, ModelSpec};

use hp_core::pipeline::Enhancer;
use hp_core::restore::{Defects, Planes};
use image::DynamicImage;

/// The neural steps for [`hp_core::pipeline::extract_photos_with`]. Either model may be
/// absent; the classical results are kept where one is missing or fails.
#[derive(Default)]
pub struct MlEnhancer {
    pub inpainter: Option<Inpainter>,
    pub colorizer: Option<Colorizer>,
    pub inpaint: InpaintParams,
}

impl MlEnhancer {
    /// Load whichever models are wanted and available.
    pub fn load(ai_fill: bool, colorize: bool) -> Self {
        let inpainter = ai_fill
            .then(Inpainter::load_default)
            .and_then(|r| r.map_err(|e| log::warn!("{e}")).ok());
        let colorizer = colorize
            .then(Colorizer::load_default)
            .and_then(|r| r.map_err(|e| log::warn!("{e}")).ok());
        Self {
            inpainter,
            colorizer,
            inpaint: InpaintParams::default(),
        }
    }
}

impl Enhancer for MlEnhancer {
    fn refill(&self, planes: &mut Planes, defects: &Defects, dpi: f64) {
        if let Some(inp) = &self.inpainter
            && let Err(e) = inp.refill(planes, defects, dpi, &self.inpaint, &mut |_, _| {})
        {
            log::warn!("AI fill failed: {e}");
        }
    }
    fn colorize(&self, img: &DynamicImage) -> Option<DynamicImage> {
        let c = self.colorizer.as_ref()?;
        c.colorize(img, 1.0)
            .map_err(|e| log::warn!("colourisation failed: {e}"))
            .ok()
    }
}
