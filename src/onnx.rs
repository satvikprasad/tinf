use crate::{
    onnx::proto::{NodeProto, ValueInfoProto},
    ops::{self, InferShape},
    tensor::{self, IntoTensor, Tensor},
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
                _ => panic!("Invalid dimensions for tensor, symbolic dimensions not supported."),
            }
        })
        .collect();

    match t {
        DataType::Float => Ok(Tensor::zeros::<f32>(shape)),
        _ => todo!(),
    }
}

fn parse_io_tensors<T: Iterator<Item = ValueInfoProto>>(
    tensors: T,
    shapes: &mut HashMap<String, Vec<usize>>,
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
                m.insert(inp.name.clone(), buffer.len());
                shapes.insert(inp.name, Vec::from(tens.shape())); // TODO(satvik): Don't
                // make this a copy
                buffer.push(tens);
            }
            _ => todo!(),
        }
    }

    Ok(())
}

fn retrieve_tens<'a>(
    buffer: &'a [Tensor],
    m: &HashMap<String, usize>,
    name: &String,
) -> Result<&'a Tensor, Box<dyn Error>> {
    let buf_idx = *m
        .get(name)
        .ok_or_else(|| format!("Failed to retrieve tensor {} while parsing", name))?;
    return Ok(&buffer[buf_idx]);
}

fn push_tens(buffer: &mut Vec<Tensor>, m: &mut HashMap<String, usize>, name: String, tens: Tensor) {
    m.insert(name, buffer.len());
    buffer.push(tens);
}

struct NodeAttributes {
    pub dilations: [usize; 2],
    pub strides: [usize; 2],
    pub group: usize,
    pub pads: [usize; 4],
    pub kernel_shape: [usize; 2],
    pub pad_type: ops::AutoPadType,
}

fn parse_attrs(attributes: Vec<proto::AttributeProto>) -> NodeAttributes {
    let mut dilations = [1usize; 2];
    let mut strides = [1usize; 2];
    let mut group = 1usize;
    let mut pads = [0usize; 4];
    let mut pad_type = ops::AutoPadType::NotSet;
    let mut kernel_shape = [1usize; 2];

    for attr in attributes {
        match attr.name.as_str() {
            "dilations" => dilations = [attr.ints[0] as usize, attr.ints[1] as usize],
            "strides" => strides = [attr.ints[0] as usize, attr.ints[1] as usize],
            "group" => group = attr.i as usize,
            "auto_pad" => {
                let val = str::from_utf8(attr.s.as_slice())
                    .expect("auto_pad attr should be valid UTF-8.");

                pad_type = match val {
                    "SAME_UPPER" => ops::AutoPadType::SameUpper,
                    "SAME_LOWER" => ops::AutoPadType::SameLower,
                    "NOTSET" => ops::AutoPadType::NotSet,
                    _ => panic!("{} pad type not implemented.", val),
                };
            }
            "pads" => {
                pads = [
                    attr.ints[0] as usize,
                    attr.ints[1] as usize,
                    attr.ints[2] as usize,
                    attr.ints[3] as usize,
                ]
            }
            "kernel_shape" => {
                kernel_shape = [attr.ints[0] as usize, attr.ints[1] as usize];
            }
            _ => continue,
        }
    }

    NodeAttributes {
        dilations,
        strides,
        group,
        pads,
        kernel_shape,
        pad_type,
    }
}

fn parse_tensor_shapes(
    node: NodeProto,
    shapes: &mut HashMap<String, Vec<usize>>,
    buffer: &Vec<Tensor>, // Required for initializer lookups,
    tensors: &HashMap<String, usize>,
) -> Result<(), Box<dyn Error>> {
    let mut input_shapes = Vec::<&[usize]>::new();

    for inp in &node.input {
        let shape = shapes.get(inp).ok_or(format!(
            "Input {} not found. Are you sure this graph was topologically sorted?",
            inp
        ))?;

        input_shapes.push(shape.as_slice());
    }

    let attrs = parse_attrs(node.attribute);

    match node.op_type.as_str() {
        "Conv" => {
            let shape = match attrs.pad_type {
                ops::AutoPadType::NotSet => {
                    let cd = ops::ConvData {
                        pads: attrs.pads,
                        strides: attrs.strides,
                        dilations: attrs.dilations,
                        group: attrs.group,
                    };

                    cd.infer_shape(input_shapes[0..2].try_into()?)
                }
                // TODO(satvik): Extract this into a seperate method
                _ => vec![
                    input_shapes[0][0],
                    input_shapes[1][0],
                    input_shapes[0][2].div_ceil(attrs.strides[0]),
                    input_shapes[0][3].div_ceil(attrs.strides[1]),
                ],
            };

            shapes.insert(
                node.output
                    .into_iter()
                    .next()
                    .ok_or_else(|| "Require at least on output for Conv")?,
                shape,
            );
        }

        "Add" => {
            let shape = ops::Op::Add.infer_shape(input_shapes[0..2].try_into()?);

            shapes.insert(
                node.output.into_iter().next().ok_or_else(|| panic!())?,
                shape,
            );
        }

        "Relu" => {
            // TODO(satvik): Verify this actually works
            // Because we're going to do the ReLU in-place, we just point output to input.
            let in_shape = shapes.get(&node.input[0]).ok_or("Required input for ReLU")?.clone();

            shapes.insert(
                node.output.into_iter().next().ok_or_else(|| panic!())?,
                in_shape, // TODO: Here we can reuse the original vec rather than
                                            // construct a new one from this slice
            );
        }

        "MaxPool" => {
            let input = input_shapes[0];

            // TODO(satvik): Extract this into a seperate function
            let shape = match attrs.pad_type {
                ops::AutoPadType::SameLower | ops::AutoPadType::SameUpper => vec![
                    input[0],
                    input[1],
                    input[2].div_ceil(attrs.strides[0]),
                    input[3].div_ceil(attrs.strides[1]),
                ],
                _ => {
                    let mpd = ops::MaxPoolData {
                        pads: attrs.pads,
                        strides: attrs.strides,
                        dilations: attrs.dilations,
                        kernel_shape: attrs.kernel_shape,
                    };

                    mpd.infer_shape(&[input])
                }
            };

            match node.output.len() {
                1usize => shapes.insert(
                    node.output.into_iter().next().ok_or_else(|| panic!())?,
                    shape,
                ),
                2usize => todo!(),
                _ => panic!("MaxPool operator has a maximum of 2 outputs (and a minimum of 1)."),
            };
        }

        "Reshape" => {
            // TODO: Verify if this logic works, but I think here we can
            // retain the tensor in memory and simply "reshape" when the node is executed
            //
            // TODO(satvik): We're going to have to split shape inference and allocation into
            // seperate passes in the future here to reduce memory overhead.

            // Lookup for value of the reshape
            let new_shape_tens = retrieve_tens(buffer, tensors, &node.input[1])?;
            let new_shape: &[i64] = new_shape_tens
                .data()
                .ok_or("Require input index 1 of Reshape to be a new shape.")?;

            shapes.insert(
                node.output
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("Required at least one output for Reshape"))?,
                new_shape.into_iter().map(|i| *i as usize).collect(),
            );
        }

        "MatMul" => {
            let shape = ops::Op::MatMul.infer_shape(input_shapes.as_slice());

            shapes.insert(
                node.output
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("Require at least one output for MatMul"))?,
                shape,
            );
        }

        op => unimplemented!("Operator {} not implemented yet.", op),
    }

    Ok(())
}

pub fn parse(path: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let proto = proto::ModelProto::decode(bytes.as_slice())?;

    let g = proto.graph.ok_or("Model did not contain a graph.")?;

    let mut buffers = Vec::<Tensor>::new();
    let mut tens_to_buffer = HashMap::<String, usize>::with_capacity(g.node.len() + g.input.len() + g.output.len());
    let mut shapes = HashMap::<String, Vec<usize>>::with_capacity(g.node.len() + g.input.len() + g.output.len());

    // Parse input/output tensors
    parse_io_tensors(
        g.input.into_iter().chain(g.output.into_iter()),
        &mut shapes,
        &mut buffers,
        &mut tens_to_buffer,
    )?;

    // Parse protos for initializers
    for initializer in g.initializer {
        parse_initializer(initializer, &mut shapes, &mut buffers, &mut tens_to_buffer)?;
    }

    // Shape inference
    for n in g.node {
        parse_tensor_shapes(n, &mut shapes, &mut buffers, &mut tens_to_buffer)?;
    }

    println!("{:#?}", shapes);

    Ok(())
}

fn parse_initializer(
    t: proto::TensorProto,
    shapes: &mut HashMap<String, Vec<usize>>,
    buffer: &mut Vec<Tensor>,
    m: &mut HashMap<String, usize>,
) -> Result<(), Box<dyn Error>> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    let dt = DataType::try_from(t.data_type)?;
    let shape: Vec<usize> = t.dims.into_iter().map(|d| d as usize).collect();

    // We're guaranteed to have never encountered this tensor before,
    // initializers are unique.
    m.insert(t.name.clone(), buffer.len());

    shapes.insert(t.name, shape.clone()); // TODO(satvik): Think about erasing this clone

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
