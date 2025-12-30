mod onnx;
mod tensor;
mod ops;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    onnx::parse("models/mnist-12.onnx").unwrap();
}
