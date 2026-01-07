#![feature(portable_simd)]

use crate::ops::matmul;

mod model;
mod onnx;
mod ops;
mod tensor;

fn main() {
    let mut model = onnx::parse("models/mnist-12.onnx").unwrap();
    let start = std::time::Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        model.execute();
    }

    let elapsed = start.elapsed();
    println!(
        "{:.2} ms/inference",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );
}
