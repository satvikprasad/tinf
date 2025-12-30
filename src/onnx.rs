use crate::{
    onnx::proto::ValueInfoProto,
    tensor::{self, Tensor},
};
use prost::Message;
use std::{collections::HashMap, error::Error, fs};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug)]
pub struct Model {
    buffers: Vec<Tensor>,
}

pub fn parse_tensor(tensor: proto::type_proto::Tensor) -> Result<Tensor, Box<dyn Error>> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    let t = DataType::try_from(tensor.elem_type)?;

    let shape_proto = tensor
        .shape
        .ok_or::<String>("Required tensor to have a shape.".into())?;
    let shape = shape_proto
        .dim
        .into_iter()
        .map(|d| {
            use proto::tensor_shape_proto::dimension::Value;

            match d.value {
                Some(Value::DimValue(v)) => v as usize,
                _ => panic!("Invalid dimensions for tensor."),
            }
        })
        .collect();

    match t {
        DataType::Float => Ok(Tensor::zeros::<f32>(shape)),
        _ => todo!(),
    }
}

fn parse_tensors<T: Iterator<Item = ValueInfoProto>>(
    tensors: T,
    buffer: &mut Vec<Tensor>,
    m: &mut HashMap<String, usize>,
) -> Result<(), Box<dyn Error>> {
    use proto::type_proto::Value;

    for inp in tensors {
        let t = inp.r#type.ok_or::<String>(
            "type field must be present for inputs and outputs of the top-level graph in an ONNX repr.".into())?;

        let val = t.value.ok_or::<String>("type field has no value.".into())?;

        match val {
            Value::TensorType(tensor) => {
                let tens = parse_tensor(tensor)?;
                m.insert(inp.name, buffer.len());
                buffer.push(tens);
            }
            _ => todo!(),
        }
    }

    Ok(())
}

pub fn parse(path: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let proto = proto::ModelProto::decode(bytes.as_slice())?;

    let g = proto.graph.ok_or("Model did not contain a graph.")?;

    let mut buffers = Vec::<Tensor>::new();
    let mut tens_to_buffer = HashMap::<String, usize>::new();

    // Parse input/output tensors
    parse_tensors(
        g.input.into_iter().chain(g.output.into_iter()),
        &mut buffers,
        &mut tens_to_buffer,
    )?;

    println!("Parsing initializers...");

    // Parse protos for initializers
    for initializer in g.initializer {
        parse_tensor_proto(initializer, &mut buffers, &mut tens_to_buffer)?;
    }

    // Parse nodes
    for n in g.node {
        for tens in n.input.into_iter().chain(n.output.into_iter()) {
            tens_to_buffer.entry(tens).or_insert_with(|| {
                // Will need to infer the shape of this tensor. Since
                // this is topologically sorted, input tensors are already present.
                todo!()
            });
        }
    }

    println!("{:#?}", tens_to_buffer);

    Ok(())
}

fn parse_tensor_proto(
    t: proto::TensorProto,
    buffer: &mut Vec<Tensor>,
    m: &mut HashMap<String, usize>,
) -> Result<(), Box<dyn Error>> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    println!("| Found {} with shape {:?}", t.name, t.dims);

    let dt = DataType::try_from(t.data_type)?;
    let shape = t.dims.into_iter().map(|d| d as usize).collect();

    // We're guaranteed to have never encountered this tensor before,
    // initializers are unique.
    m.insert(t.name, buffer.len());

    let tens = match dt {
        DataType::Float => Tensor::new(t.float_data, shape),
        DataType::Int64 => Tensor::new(t.int64_data, shape),
        _ => {
            println!("{:#?}", dt);
            todo!();
        }
    };

    buffer.push(tens); 
    Ok(())
}
