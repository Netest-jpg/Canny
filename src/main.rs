use image::{DynamicImage, GrayImage, ImageReader};
use std::collections::VecDeque;
use std::error::Error;

#[inline]
fn convolve3x3(src: &[f32], width: usize, height: usize, kernel: &[f32; 9]) -> Vec<f32> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            for ky in 0..3 {
                for kx in 0..3 {
                    total += kernel[ky * 3 + kx] * src[(y + ky) * pw + (x + kx)];
                }
            }
            slate.push(total);
        }
    }
    slate
}

#[inline]
fn to_image(src: &[f32], width: u32, height: u32) -> DynamicImage {
    let pixels: Vec<u8> = src.iter().map(|&p| p as u8).collect();

    let gray =
        GrayImage::from_raw(width, height, pixels).expect("Vector must be of len: width * height");

    DynamicImage::ImageLuma8(gray)
}

#[inline]
fn zero_padding(src: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(pw * (height + 2));

    slate.resize(pw, 0.0);

    for row in src.chunks_exact(width) {
        slate.push(0.0);
        slate.extend_from_slice(row);
        slate.push(0.0);
    }

    slate.resize(pw * (height + 2), 0.0);

    slate
}

#[inline]
fn edge_padding(src: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(pw * (height + 2));

    for y in 0..height + 2 {
        let sy = y.saturating_sub(1).min(height - 1);
        let row = &src[sy * width..(sy + 1) * width];

        slate.push(row[0]);
        slate.extend_from_slice(row);
        slate.push(row[width - 1]);
    }

    slate
}

#[inline]
fn remove_pad(src: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(width * height);

    for i in 0..height {
        let start = (i + 1) * pw + 1;
        slate.extend_from_slice(&src[start..start + width]);
    }

    slate
}

fn main() -> Result<(), Box<dyn Error>> {
    let image = ImageReader::open("Mona_Lisa.jpg")?.decode()?;
    #[rustfmt::skip]
    let gx_kernel = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    ];
    #[rustfmt::skip]
    let gy_kernel = [
        -1.0, -2.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 1.0
    ];

    let gray = image.to_luma8();
    let gray = image::imageops::blur(&gray, 5.0);
    let (height, width) = (gray.height() as usize, gray.width() as usize);
    let src: Vec<f32> = gray.as_raw().iter().map(|&p| p as f32).collect();

    let padded = edge_padding(&src, width, height);
    let gx = convolve3x3(&padded, width, height, &gx_kernel);
    let gy = convolve3x3(&padded, width, height, &gy_kernel);

    let mut magnitude: Vec<f32> = Vec::with_capacity(width * height);
    let mut direction: Vec<f32> = Vec::with_capacity(width * height);
    for (&x, &y) in gx.iter().zip(gy.iter()) {
        let g: f32 = (x * x + y * y).sqrt();
        let mut theta = y.atan2(x).to_degrees();
        if theta < 0.0 {
            theta += 180.0;
        }
        let theta = if theta < 22.5 || theta >= 157.5 {
            0.0
        } else if theta < 67.5 {
            45.0
        } else if theta < 112.5 {
            90.0
        } else {
            135.0
        };
        magnitude.push(g);
        direction.push(theta);
    }

    let padded_width = width + 2;
    let magnitude = zero_padding(&magnitude, width, height);
    let direction = zero_padding(&direction, width, height);
    let mut suppressed = vec![0.0f32; magnitude.len()];
    for y in 0..height {
        for x in 0..width {
            let i = (y + 1) * padded_width + (x + 1);
            let (a, b) = match direction[i] as u32 {
                0 => (i - 1, i + 1),
                45 => (i - padded_width - 1, i + padded_width + 1),
                90 => (i - padded_width, i + padded_width),
                _ => (i - padded_width + 1, i + padded_width - 1),
            };
            let m = magnitude[i];
            if m >= magnitude[a] && m >= magnitude[b] {
                suppressed[i] = m;
            }
        }
    }
    let mut magnitude = suppressed;

    let max_mag = magnitude.iter().cloned().fold(0.0f32, f32::max);
    let t_high = 0.20 * max_mag;
    let t_low = 0.40 * t_high;
    let mut strong_edges: VecDeque<usize> = VecDeque::new();
    for (i, magnitude) in magnitude.iter_mut().enumerate() {
        if *magnitude >= t_high {
            *magnitude = 255.0;
            strong_edges.push_back(i);
        } else if *magnitude < t_low {
            *magnitude = 0.0;
        } else {
            *magnitude = 127.0;
        }
    }

    while let Some(i) = strong_edges.pop_front() {
        for dy in [-1isize, 0, 1] {
            for dx in [-1isize, 0, 1] {
                let j = (i as isize + dy * padded_width as isize + dx) as usize;
                if magnitude[j] == 127.0 {
                    magnitude[j] = 255.0;
                    strong_edges.push_back(j);
                }
            }
        }
    }
    for magnitude in &mut magnitude {
        if *magnitude == 127.0 {
            *magnitude = 0.0;
        }
    }

    let magnitude = remove_pad(&magnitude, width, height);
    let image = to_image(&magnitude, width as u32, height as u32);
    image.save("out.png").unwrap();
    Ok(())
}
