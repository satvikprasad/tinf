use crate::{
    onnx::proto::{NodeProto, ValueInfoProto},
    ops::{self, InferShape},
    tensor::{self, IntoTensor, Tensor},
};
use core::num;
use prost::{Message, bytes::buf};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs,
    hash::Hash,
};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug)]
pub struct Model {
    buffers: Vec<Tensor>,
}

#[derive(Debug)]
pub struct Node {
    op: ops::Op,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
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

fn parse_attrs(attributes: &[proto::AttributeProto]) -> NodeAttributes {
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

// TODO(satvik): Maybe think of removing the allocations here?
fn parse_tensor_shapes(node: &Node, shapes: &mut [Vec<usize>]) -> Result<(), Box<dyn Error>> {
    let in0: Vec<usize> = shapes[node.inputs[0]].clone();

    match &node.op {
        ops::Op::ReLU | ops::Op::MaxPool(_) | ops::Op::Reshape(_) => {
            node.op.infer_shape(&[&in0], &mut shapes[node.outputs[0]]);
        }
        ops::Op::MatMul | ops::Op::Add | ops::Op::Conv(_) => {
            let in1: Vec<usize> = shapes[node.inputs[1]].clone();
            node.op
                .infer_shape(&[&in0, &in1], &mut shapes[node.outputs[0]]);
        }
    };
    Ok(())
}

/**
 * Generates tensor mappings and fills parsed nodes' inputs and outputs.
 *
 * @param in_nodes Input node protos, preallocated
 * @param tensor_mapping Mapping from tensor names to their logical index, preallocated and to be
 * filled.
 * @param num_tensors Mutable reference to variable storing the number of tensors, to be filled.
 * @param out_nodes Logical nodes to be parsed
 */
fn parse_nodes_and_generate_tensor_mappings(
    in_nodes: &[NodeProto],
    tensor_mapping: &mut HashMap<String, usize>,
    num_tensors: &mut usize,
    out_nodes: &mut [Node],
) {
    *num_tensors = 0usize;

    for (i, n) in in_nodes.iter().enumerate() {
        for inp in &n.input {
            // TODO(satvik): Is this alloc adding overhead?
            let en = tensor_mapping.entry(inp.to_string()).or_insert_with(|| {
                *num_tensors += 1;
                *num_tensors - 1
            });

            out_nodes[i].inputs.push(*en);
        }

        for out in &n.output {
            let en = tensor_mapping.entry(out.to_string()).or_insert_with(|| {
                *num_tensors += 1;
                *num_tensors - 1
            });

            out_nodes[i].outputs.push(*en);
        }
    }
}

fn parse_input_shapes(
    inputs: Vec<ValueInfoProto>,
    tensor_mapping: &mut HashMap<String, usize>,
    shapes: &mut Vec<Vec<usize>>,
) -> Result<(), Box<dyn Error>> {
    // Parse input tensor shapes, filling shapes
    for inp in inputs {
        let buf_idx = *tensor_mapping
            .get(&inp.name)
            .ok_or_else(|| format!("No operation uses input tensor {} as input", inp.name))?;

        let input_type = inp
            .r#type
            .ok_or_else(|| format!("Expected type of input {}", inp.name))?;

        use proto::tensor_shape_proto::dimension::Value::DimValue;
        use proto::type_proto::Value;

        match input_type.value {
            Some(Value::TensorType(tensor)) => {
                let shape_proto = tensor.shape.ok_or_else(|| "Tensor shape expected.")?;
                let shape: Vec<usize> = shape_proto
                    .dim
                    .iter()
                    .map(|d| match d.value {
                        Some(DimValue(v)) => v as usize,
                        _ => panic!(),
                    })
                    .collect();

                shapes[buf_idx] = shape;
            }
            _ => todo!("{:#?} not supported", input_type.value),
        };
    }

    Ok(())
}

pub fn generate_buffer_mappings(
    nodes: &[Node],
    buffers: &mut [usize],
    tensor_size: usize,
    tensor_mapping: &mut HashMap<String, usize>,
    ins: &[String],
    outs: &[String],
) -> Result<(), Box<dyn Error>> {
    // Map ending node to tensors
    let mut terminations = vec![Vec::<usize>::new(); nodes.len() + 2];
    let mut adjacencies = vec![Vec::<usize>::new(); tensor_size];

    // Should track the start node this tensor data is required, and end node
    let mut liveness: Vec<(usize, usize)> = vec![(0, 0); tensor_size];
    let retr = |tens: &String| -> Result<usize, Box<dyn Error>> {
        let tens_idx = *tensor_mapping
            .get(tens)
            .ok_or_else(|| format!("Tensor {} not found", tens))?;

        Ok(tens_idx)
    }; // TODO(satvik): Verify this indirection doesn't add extra overhead.

    // All initializers and inputs are created at the start of inference.
    for tens in ins {
        liveness[retr(tens)?].0 = 0; // Tensor is created at the start of inference
    }

    // All outputs are consumed at the end of inference.
    for tens in outs {
        liveness[retr(tens)?].1 = nodes.len() + 1; // Tensor is created at the start of inference
    }

    // We need to map tensors to buffers, by first generating tensor liveness.
    for (i, node) in nodes.iter().enumerate() {
        for &inp in &node.inputs {
            liveness[inp].1 = i; // Must be consumed earliest this node.
        }

        for &out in &node.outputs {
            liveness[out].0 = i; // Must be created latest this node.
        }
    }

    for (tens_idx, &(_, b)) in liveness.iter().enumerate() {
        terminations[b].push(tens_idx);
    }

    for (i, &(a, b)) in liveness.iter().enumerate() {
        for (j, &(p, q)) in liveness.iter().enumerate() {
            if (p <= b && q >= a) || (q >= a && p <= b) {
                adjacencies[i].push(j);
                adjacencies[j].push(i);
            }
        }
    }

    for tens_idx in 0..tensor_size {
        let mut adj_bufs = HashSet::with_capacity(adjacencies[tens_idx].len());

        for &nbor in &adjacencies[tens_idx] {
            adj_bufs.insert(buffers[nbor]);
        }

        let mut buf_idx = 0;
        while adj_bufs.contains(&buf_idx) {
            buf_idx += 1;
        }

        buffers[tens_idx] = buf_idx;
    }

    Ok(())
}

fn parse_op(node: &NodeProto, initializers: &HashMap<String, Tensor>) -> ops::Op {
    let attrs = parse_attrs(node.attribute.as_slice());

    match node.op_type.as_str() {
        "Add" => ops::Op::Add,
        "Conv" => {
            let cd = ops::ConvData {
                pads: attrs.pads,
                strides: attrs.strides,
                dilations: attrs.dilations,
                group: attrs.group,
                pad_type: attrs.pad_type,
            };

            ops::Op::Conv(cd)
        }
        "MatMul" => ops::Op::MatMul,
        "Relu" => ops::Op::ReLU,
        "MaxPool" => {
            let mpd = ops::MaxPoolData {
                pads: attrs.pads,
                strides: attrs.strides,
                dilations: attrs.dilations,
                kernel_shape: attrs.kernel_shape,
                pad_type: attrs.pad_type,
            };

            ops::Op::MaxPool(mpd)
        }
        "Reshape" => {
            let shape_tensor = &initializers[&node.input[1]];

            match shape_tensor {
                Tensor::I64(tensor_data) => ops::Op::Reshape(ops::ReshapeData {
                    output_shape: tensor_data.data.iter().map(|&v| v as usize).collect(),
                }),
                _ => panic!(),
            }
        }
        op => todo!("{} op not supported", op),
    }
}

pub fn parse(path: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let proto = proto::ModelProto::decode(bytes.as_slice())?;

    let g = proto.graph.ok_or("Model did not contain a graph.")?;

    let mut tensor_size = 0usize;
    let mut tensor_mapping =
        HashMap::<String, usize>::with_capacity(g.node.len() + g.input.len() + g.output.len());

    let mut ins = Vec::<String>::with_capacity(g.input.len() + g.initializer.len());

    let mut initializers = HashMap::<String, Tensor>::with_capacity(g.initializer.len());
    for init in g.initializer {
        let (name, tens) = parse_initializer(init)?;

        initializers.insert(name.clone(), tens);
        ins.push(name);
    }

    let outs: Vec<String> = g.output.into_iter().map(|v| v.name).collect();

    let mut nodes: Vec<Node> = g
        .node
        .iter()
        .map(|n| Node {
            inputs: Vec::with_capacity(n.input.len()),
            outputs: Vec::with_capacity(n.output.len()),
            op: parse_op(n, &initializers), // TODO(satvik): Parse the operation here properly
        })
        .collect(); // Pre-allocate space for nodes.

    parse_nodes_and_generate_tensor_mappings(
        &g.node,
        &mut tensor_mapping,
        &mut tensor_size,
        &mut nodes,
    );

    let mut buffers = vec![0usize; tensor_size];
    generate_buffer_mappings(
        nodes.as_slice(),
        buffers.as_mut_slice(),
        tensor_size,
        &mut tensor_mapping,
        ins.as_slice(),
        outs.as_slice(),
    )?;

    // We've generated every buffer. We can now perform shape inference on each tensor,
    // and then determine max sizes of tensors in each buffer.

    println!("{:#?}", buffers);

    let mut shapes = vec![Vec::new(); tensor_size]; // Shapes for each in/out tensor
    // parse_input_shapes(g.input, &mut tensor_mapping, &mut shapes)?;
    for inp in g.input {
        let input_type = inp
            .r#type
            .ok_or_else(|| format!("Expected graph input {} to have type", inp.name))?;

        let tens_idx = *tensor_mapping.get(&inp.name).ok_or_else(|| format!(""))?;

        match input_type.value {
            Some(proto::type_proto::Value::TensorType(tensor)) => {
                use proto::tensor_shape_proto::dimension::Value::DimValue;
                let shape_proto = tensor.shape.ok_or_else(|| "Tensor shape expected.")?;

                shapes[tens_idx].extend(shape_proto.dim.iter().map(|d| match d.value {
                    Some(DimValue(v)) => v as usize,
                    _ => panic!(),
                }));
            }
            _ => todo!("{:#?} not supported", input_type.value),
        }

        ins.push(inp.name);
    }

    for (tensor_name, tens) in &initializers {
        let tens_idx = *tensor_mapping.get(tensor_name).ok_or_else(|| format!(""))?;

        shapes[tens_idx].extend(tens.shape().iter().map(|&v| v));
    }

    for node in &nodes {
        parse_tensor_shapes(node, shapes.as_mut_slice());
    }

    println!("{:#?} {}", shapes, shapes.len());
    println!("{:#?}", tensor_mapping);

    Ok(())
}

/**
 * Performs no allocations for tensor data.
 */
fn parse_initializer(t: proto::TensorProto) -> Result<(String, Tensor), Box<dyn Error>> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    let dt = DataType::try_from(t.data_type)?;
    let shape: Vec<usize> = t.dims.into_iter().map(|d| d as usize).collect();

    match dt {
        DataType::Float => Ok((t.name, Tensor::new(t.float_data, shape))),
        DataType::Int64 => Ok((t.name, Tensor::new(t.int64_data, shape))),
        _ => {
            println!("{:#?}", dt);
            todo!();
        }
    }
}
