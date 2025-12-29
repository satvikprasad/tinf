mod onnx;
mod tensor;

fn main() {
    onnx::parse("models/mnist-12.onnx").unwrap();
}
