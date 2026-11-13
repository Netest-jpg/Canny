use image::{GrayImage, ImageReader};
use std::collections::VecDeque;
use std::error::Error;

#[inline]
fn calculate_gradients(
    src: &[u8],
    width: usize,
    height: usize,
    gx_kernel: &[i8; 9],
    gy_kernel: &[i8; 9],
) -> (Vec<u32>, Vec<u8>) {
    let pw = width + 2;
    let mut magnitude = Vec::with_capacity(width * height);
    let mut direction = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let mut gx = 0i16;
            let mut gy = 0i16;

            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = src[(y + ky) * pw + (x + kx)] as i16;
                    let k = ky * 3 + kx;

                    gx += gx_kernel[k] as i16 * pixel;
                    gy += gy_kernel[k] as i16 * pixel;
                }
            }

            let gx = gx as i32;
            let gy = gy as i32;

            let g = (gx * gx + gy * gy) as u32;
            let mut theta = (gy as f32).atan2(gx as f32).to_degrees();
            if theta < 0.0 {
                theta += 180.0;
            }
            let theta = if theta < 22.5 || theta >= 157.5 {
                0
            } else if theta < 67.5 {
                45
            } else if theta < 112.5 {
                90
            } else {
                135
            };

            magnitude.push(g);
            direction.push(theta);
        }
    }

    (magnitude, direction)
}

#[inline]
fn to_image(src: Vec<u8>, width: u32, height: u32) -> GrayImage {
    GrayImage::from_raw(width, height, src).unwrap()
}

#[inline]
fn zero_padding_u32(src: &[u32], width: usize, height: usize) -> Vec<u32> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(pw * (height + 2));
    slate.resize(pw, 0);
    for row in src.chunks_exact(width) {
        slate.push(0);
        slate.extend_from_slice(row);
        slate.push(0);
    }
    slate.resize(pw * (height + 2), 0);
    slate
}

#[inline]
fn zero_padding_u8(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let pw = width + 2;
    let mut slate = Vec::with_capacity(pw * (height + 2));
    slate.resize(pw, 0);
    for row in src.chunks_exact(width) {
        slate.push(0);
        slate.extend_from_slice(row);
        slate.push(0);
    }
    slate.resize(pw * (height + 2), 0);
    slate
}

#[inline]
fn edge_padding(src: &[u8], width: usize, height: usize) -> Vec<u8> {
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
fn remove_pad(src: &[u8], width: usize, height: usize) -> Vec<u8> {
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
    let gx_kernel : [i8; 9]= [
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ];
    #[rustfmt::skip]
    let gy_kernel: [i8; 9] = [
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    ];

    let gray = image.to_luma8();
    let gray = image::imageops::blur(&gray, 5.0);
    let (height, width) = (gray.height() as usize, gray.width() as usize);

    let padded = edge_padding(gray.as_raw(), width, height);
    let (raw_magnitude, raw_direction) =
        calculate_gradients(&padded, width, height, &gx_kernel, &gy_kernel);

    let pw = width + 2;
    let magnitude = zero_padding_u32(&raw_magnitude, width, height);
    let direction = zero_padding_u8(&raw_direction, width, height);
    let mut suppressed = vec![0u32; magnitude.len()];
    for y in 0..height {
        for x in 0..width {
            let i = (y + 1) * pw + (x + 1);
            let (a, b) = match direction[i] {
                0 => (i - 1, i + 1),
                45 => (i - pw - 1, i + pw + 1),
                90 => (i - pw, i + pw),
                _ => (i - pw + 1, i + pw - 1),
            };
            let m = magnitude[i];
            if m >= magnitude[a] && m >= magnitude[b] {
                suppressed[i] = m;
            }
        }
    }

    let max_mag = *suppressed.iter().max().unwrap_or(&0);
    let mut edges: Vec<u8> = Vec::with_capacity(suppressed.len());

    if max_mag == 0 {
        edges.resize(suppressed.len(), 0);
    } else {
        let t_high = max_mag as f64 * 0.04;
        let t_low = t_high * 0.16;

        for &magnitude in &suppressed {
            let value = magnitude as f64;
            if value >= t_high {
                edges.push(255);
            } else if value < t_low {
                edges.push(0);
            } else {
                edges.push(127);
            }
        }
    }

    let mut queue = VecDeque::new();
    for (i, &pixel) in edges.iter().enumerate() {
        if pixel == 255 {
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for dy in [-1isize, 0, 1] {
            for dx in [-1isize, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let j = (i as isize + dy * pw as isize + dx) as usize;
                if edges[j] == 127 {
                    edges[j] = 255;
                    queue.push_back(j);
                }
            }
        }
    }
    for pixel in &mut edges {
        if *pixel == 127 {
            *pixel = 0;
        }
    }

    let edges = remove_pad(&edges, width, height);
    let image = to_image(edges, width as u32, height as u32);
    image.save("out.png")?;
    Ok(())
}
