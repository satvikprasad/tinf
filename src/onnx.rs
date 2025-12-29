use crate::tensor::{self, Tensor};
use prost::Message;
use std::{error::Error, fs};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub fn parse(path: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let proto = proto::ModelProto::decode(bytes.as_slice())?;

    match proto.graph {
        Some(g) => {
            let inputs: String = g.input.iter().map(|i| i.name.clone()).collect();

            let outputs: String = g.output.iter().map(|o| o.name.clone()).collect();

            let initializers: Vec<Tensor> = g
                .initializer
                .into_iter()
                .filter_map(|t| parse_tensor(t).ok())
                .collect();

            println!("{:#?}", initializers);

            Ok(())
        }
        None => Err("Model did not contain a graph.".into()),
    }
}

pub fn parse_tensor(t: proto::TensorProto) -> Result<Tensor, Box<dyn Error>> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    let dt = DataType::try_from(t.data_type)?;
    let shape = t.dims.into_iter().map(|d| d as usize).collect();

    match dt {
        DataType::Float => {
            Ok(Tensor::new(t.float_data, shape))
        },
        DataType::Int64 => {
            Ok(Tensor::new(t.int64_data, shape))
        }
        _ => {
            println!("{:#?}", dt); todo!();
        }
    }
}