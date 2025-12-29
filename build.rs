use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["protobuf/onnx.proto3"], &["protobuf/"])?;
    Ok(())
}