//! Video output generation module for LTX-Video inference results.
//! Supports MP4 format with H.264 codec and PNG frame export.

use candle::{IndexOp, Result, Tensor};
use image::{ImageBuffer, Rgb};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Video output format options
#[derive(Clone, Debug, Copy)]
pub enum VideoFormat {
    /// MP4 video format (requires ffmpeg)
    Mp4,
    /// Individual PNG frames
    Frames,
    /// Both MP4 and frames
    Both,
}

/// Video output configuration
#[derive(Clone, Debug)]
pub struct VideoOutputConfig {
    /// Output video format
    pub format: VideoFormat,
    /// Frame rate in FPS (default: 25)
    pub fps: u32,
    /// Output directory for frames (if saving individual frames)
    pub frames_dir: Option<String>,
}

impl Default for VideoOutputConfig {
    fn default() -> Self {
        Self {
            format: VideoFormat::Frames,
            fps: 25,
            frames_dir: None,
        }
    }
}

/// Result of video generation
pub struct VideoGenerationResult {
    /// Path to generated MP4 file (if format includes MP4)
    pub mp4_path: Option<String>,
    /// Path to frames directory (if format includes frames)
    pub frames_dir: Option<String>,
    /// Number of frames generated
    pub num_frames: usize,
    /// Video height
    pub height: usize,
    /// Video width
    pub width: usize,
    /// Frames per second
    pub fps: u32,
}

/// Convert tensor to video frames
///
/// Assumes tensor is in shape (B, C, F, H, W) with values in range [0, 255]
/// Returns Vec of (F, H, W) tensors, one per frame
pub fn tensor_to_frames(video_tensor: &Tensor) -> Result<Vec<Tensor>> {
    let dims = video_tensor.dims();
    if dims.len() != 5 {
        return Err(candle::Error::Msg(format!(
            "Expected 5D tensor (B, C, F, H, W), got {}D",
            dims.len()
        )));
    }

    let batch_size = dims[0];
    let channels = dims[1];
    let num_frames = dims[2];
    let _height = dims[3];
    let _width = dims[4];

    if channels != 3 {
        return Err(candle::Error::Msg(format!(
            "Expected 3 channels (RGB), got {}",
            channels
        )));
    }

    if batch_size != 1 {
        return Err(candle::Error::Msg(format!(
            "Expected batch size 1, got {}",
            batch_size
        )));
    }

    let mut frames = Vec::new();

    // Extract each frame
    for frame_idx in 0..num_frames {
        // Extract frame: video_tensor[0, :, frame_idx, :, :]
        let frame = video_tensor
            .i(0)? // Remove batch dimension
            .i((.., frame_idx, .., ..))?; // Get frame

        frames.push(frame);
    }

    Ok(frames)
}

/// Convert a single frame tensor to RGB image buffer
///
/// Frame should be in shape (C, H, W) with values in range [0, 255]
fn frame_to_image(frame: &Tensor) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let dims = frame.dims();
    if dims.len() != 3 {
        return Err(candle::Error::Msg(format!(
            "Expected 3D tensor (C, H, W), got {}D",
            dims.len()
        )));
    }

    let channels = dims[0];
    let height = dims[1];
    let width = dims[2];

    if channels != 3 {
        return Err(candle::Error::Msg(format!(
            "Expected 3 channels (RGB), got {}",
            channels
        )));
    }

    // Convert tensor to u8 data
    let data = frame.flatten_all()?.to_vec1::<f32>()?;

    // Create image buffer
    let mut img_data = Vec::with_capacity(height * width * 3);
    for pixel_idx in 0..(height * width) {
        // Get RGB values for this pixel
        let r = data[pixel_idx].clamp(0.0, 255.0) as u8;
        let g = data[height * width + pixel_idx].clamp(0.0, 255.0) as u8;
        let b = data[2 * height * width + pixel_idx].clamp(0.0, 255.0) as u8;

        img_data.push(r);
        img_data.push(g);
        img_data.push(b);
    }

    let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width as u32, height as u32, img_data)
        .ok_or_else(|| candle::Error::Msg("Failed to create image buffer".to_string()))?;

    Ok(img)
}

/// Save video frames as PNG images
///
/// # Arguments
/// * `frames` - Vector of frame tensors in (C, H, W) format with values [0, 255]
/// * `output_dir` - Directory to save frames
/// * `frame_name_prefix` - Prefix for frame filenames (e.g., "frame" produces "frame_0000.png")
///
/// # Returns
/// Path to the frames directory
pub fn save_frames_as_png(
    frames: &[Tensor],
    output_dir: impl AsRef<Path>,
    frame_name_prefix: &str,
) -> Result<String> {
    let output_dir = output_dir.as_ref();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)
        .map_err(|e| candle::Error::Msg(format!("Failed to create directory: {}", e)))?;

    // Save each frame
    for (idx, frame) in frames.iter().enumerate() {
        let img = frame_to_image(frame)?;
        let filename = format!("{}_{:04}.png", frame_name_prefix, idx);
        let path = output_dir.join(&filename);

        img.save(&path)
            .map_err(|e| candle::Error::Msg(format!("Failed to save frame: {}", e)))?;
    }

    let frames_dir = output_dir
        .to_str()
        .ok_or_else(|| candle::Error::Msg("Invalid directory path".to_string()))?
        .to_string();

    println!("Saved {} frames to {}", frames.len(), frames_dir);

    Ok(frames_dir)
}

/// Save video as MP4 file using pure Rust (no ffmpeg required)
///
/// # Arguments
/// * `frames` - Vector of frame tensors in (C, H, W) format with values [0, 255]
/// * `output_path` - Path where MP4 file will be saved
/// * `fps` - Frames per second
///
/// # Returns
/// Path to the generated MP4 file
pub fn save_video_as_mp4(
    frames: &[Tensor],
    output_path: impl AsRef<Path>,
    fps: u32,
) -> Result<String> {
    use mp4::{AvcConfig, MediaConfig, Mp4Config, Mp4Writer, TrackConfig};

    if frames.is_empty() {
        return Err(candle::Error::Msg("No frames to save".to_string()));
    }

    let output_path = output_path.as_ref();

    // Get frame dimensions from first frame
    let dims = frames[0].dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(candle::Error::Msg("Invalid frame dimensions".to_string()));
    }

    let height = dims[1] as u32;
    let width = dims[2] as u32;

    // Ensure dimensions are even (required for H.264)
    let width_adjusted = if width.is_multiple_of(2) {
        width
    } else {
        width - 1
    };
    let height_adjusted = if height.is_multiple_of(2) {
        height
    } else {
        height - 1
    };

    println!(
        "Preparing to encode video: {}x{}@{}fps, {} frames",
        width_adjusted,
        height_adjusted,
        fps,
        frames.len()
    );

    // Create output file with buffering
    let file = File::create(output_path)
        .map_err(|e| candle::Error::Msg(format!("Failed to create output file: {}", e)))?;
    let writer = BufWriter::new(file);

    // Configure MP4
    let config = Mp4Config {
        major_brand: "isom"
            .parse()
            .map_err(|_| candle::Error::Msg("Failed to parse major_brand".to_string()))?,
        minor_version: 512,
        compatible_brands: vec![
            "isom"
                .parse()
                .map_err(|_| candle::Error::Msg("Failed to parse isom brand".to_string()))?,
            "iso2"
                .parse()
                .map_err(|_| candle::Error::Msg("Failed to parse iso2 brand".to_string()))?,
            "avc1"
                .parse()
                .map_err(|_| candle::Error::Msg("Failed to parse avc1 brand".to_string()))?,
            "mp41"
                .parse()
                .map_err(|_| candle::Error::Msg("Failed to parse mp41 brand".to_string()))?,
        ],
        timescale: fps,
    };

    let mut mp4_writer = Mp4Writer::write_start(writer, &config)
        .map_err(|e| candle::Error::Msg(format!("Failed to start MP4 writer: {}", e)))?;

    // Configure H.264 track with minimal SPS/PPS
    let track_config = TrackConfig {
        track_type: mp4::TrackType::Video,
        timescale: fps,
        language: "und".into(),
        media_conf: MediaConfig::AvcConfig(AvcConfig {
            width: width_adjusted as u16,
            height: height_adjusted as u16,
            // Minimal valid SPS (Sequence Parameter Set)
            seq_param_set: vec![
                0x27, 0x42, 0x80, 0x1e, 0x13, 0xf2, 0xe0, 0x02, 0x40, 0x00, 0x00, 0x00,
            ],
            // Minimal valid PPS (Picture Parameter Set)
            pic_param_set: vec![0x28, 0xce, 0x3c, 0x80],
        }),
    };

    mp4_writer
        .add_track(&track_config)
        .map_err(|e| candle::Error::Msg(format!("Failed to add video track: {}", e)))?;

    println!("Encoding video frames...");

    // Process each frame - convert to YUV420 for H.264
    let total_frames = frames.len();
    for (frame_idx, frame) in frames.iter().enumerate() {
        if frame_idx % 10 == 0 {
            println!("  Frame {}/{}", frame_idx, total_frames);
        }

        let img = frame_to_image(frame)?;

        // Convert RGB to YUV420 format for H.264
        let yuv420_data = rgb_to_yuv420(&img, width_adjusted as usize, height_adjusted as usize)?;

        // Create sample
        let sample = mp4::Mp4Sample {
            start_time: frame_idx as u64,
            duration: 1,
            rendering_offset: 0,
            is_sync: frame_idx == 0 || frame_idx % 30 == 0, // Keyframe every 30 frames
            bytes: yuv420_data.into(),
        };

        // Write sample to track 1
        mp4_writer
            .write_sample(1, &sample)
            .map_err(|e| candle::Error::Msg(format!("Failed to write sample: {}", e)))?;
    }

    // Finalize MP4 file
    mp4_writer
        .write_end()
        .map_err(|e| candle::Error::Msg(format!("Failed to finalize MP4: {}", e)))?;

    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| candle::Error::Msg("Invalid output path".to_string()))?
        .to_string();

    println!("Video saved to: {}", output_path_str);

    Ok(output_path_str)
}

/// Convert RGB image to YUV420 format for H.264
///
/// YUV420 is the standard format for video compression
fn rgb_to_yuv420(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    if img.width() as usize != width || img.height() as usize != height {
        return Err(candle::Error::Msg(format!(
            "Image dimensions {}x{} don't match expected {}x{}",
            img.width(),
            img.height(),
            width,
            height
        )));
    }

    let mut yuv420 = Vec::with_capacity((width * height * 3) / 2);

    // Y plane (luminance)
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // BT.709 luma
            let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) as u8;
            yuv420.push(luma);
        }
    }

    // U plane (chroma) - subsampled 2x2
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let pixel = img.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // BT.709 chroma
            let u = ((b - (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 1.8556) as i16;
            let u = (u + 128).clamp(0, 255) as u8;
            yuv420.push(u);
        }
    }

    // V plane (chroma) - subsampled 2x2
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let pixel = img.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // BT.709 chroma
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let v = ((r - luma) / 1.5748) as i16;
            let v = (v + 128).clamp(0, 255) as u8;
            yuv420.push(v);
        }
    }

    Ok(yuv420)
}

/// Generate and save video output with optional MP4 and PNG frame export.
///
/// # Arguments
/// * `video_tensor` - Tensor in (B, C, F, H, W) format with values [0, 255]
/// * `output_path` - Base path for output (without extension)
/// * `config` - Output configuration
///
/// # Returns
/// Information about the generated files
pub fn generate_video_output(
    video_tensor: &Tensor,
    output_path: impl AsRef<Path>,
    config: &VideoOutputConfig,
) -> Result<VideoGenerationResult> {
    let output_path = output_path.as_ref();

    // Convert tensor to frames
    let frames = tensor_to_frames(video_tensor)?;
    let num_frames = frames.len();

    // Get dimensions from first frame
    let dims = frames[0].dims();
    let height = dims[1];
    let width = dims[2];

    let mut mp4_path = None;
    let mut frames_dir = None;

    // Save frames as PNG if requested
    if matches!(config.format, VideoFormat::Frames | VideoFormat::Both) {
        let frames_output_dir = config.frames_dir.clone().unwrap_or_else(|| {
            format!(
                "{}_frames",
                output_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("video")
            )
        });

        frames_dir = Some(save_frames_as_png(&frames, frames_output_dir, "frame")?);
    }

    // Save as MP4 if requested
    if matches!(config.format, VideoFormat::Mp4 | VideoFormat::Both) {
        let mp4_output = output_path.with_extension("mp4");
        mp4_path = Some(save_video_as_mp4(&frames, &mp4_output, config.fps)?);
    }

    Ok(VideoGenerationResult {
        mp4_path,
        frames_dir,
        num_frames,
        height,
        width,
        fps: config.fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_output_config_default() {
        let config = VideoOutputConfig::default();
        assert_eq!(config.fps, 25);
    }

    #[test]
    fn test_video_format_selection() {
        let config_mp4 = VideoOutputConfig {
            format: VideoFormat::Mp4,
            ..Default::default()
        };
        assert!(matches!(config_mp4.format, VideoFormat::Mp4));

        let config_both = VideoOutputConfig {
            format: VideoFormat::Both,
            ..Default::default()
        };
        assert!(matches!(config_both.format, VideoFormat::Both));
    }
}
