use crate::{
    model::{Model, Node},
    onnx::proto::{NodeProto, ValueInfoProto},
    ops::{self, InferShape},
    tensor::{MAX_RANK, Tensor, TensorShape, TensorType},
};
use prost::Message;
use std::{
    alloc::{Layout, alloc_zeroed},
    collections::{HashMap, HashSet},
    fs,
};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
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

/**
 * Parses a node's output tensor shape given initialized input shapes.
 */
fn parse_tensor_shapes(node: &Node, shapes: &mut [TensorShape], types: &mut [TensorType]) {
    let in0: TensorShape = shapes[node.inputs[0]]; // TODO(satvik): Allocate this on a
    // stack with a max rank.

    match &node.op {
        ops::Op::ReLU | ops::Op::MaxPool(_) | ops::Op::Reshape(_) => {
            node.op.infer_shape(&[in0], &mut shapes[node.outputs[0]]);
        }
        ops::Op::MatMul | ops::Op::Add | ops::Op::Conv(_) => {
            let in1: TensorShape = shapes[node.inputs[1]];
            node.op
                .infer_shape(&[in0, in1], &mut shapes[node.outputs[0]]);
        }
    };

    // Ouptut type is the first input type for all the supported operations.
    types[node.outputs[0]] = types[node.inputs[0]];
}

/**
 * Generates tensor mappings and fills parsed nodes' inputs and outputs.
 *
 * @param node_protos Input node protos, preallocated
 * @param tensor_mapping Mapping from tensor names to their logical index, preallocated and to be
 * filled.
 * @param num_tensors Mutable reference to variable storing the number of tensors, to be filled.
 * @param nodes Logical nodes to be parsed
 */
fn parse_nodes_and_generate_tensor_mappings(
    node_protos: &[NodeProto],
    nodes: &mut [Node],
    tensor_mapping: &mut HashMap<String, usize>,
    num_tensors: &mut usize,
) {
    *num_tensors = 0usize;

    for (i, n) in node_protos.iter().enumerate() {
        for inp in &n.input {
            // TODO(satvik): Is this alloc adding overhead?
            let en = tensor_mapping.entry(inp.to_string()).or_insert_with(|| {
                *num_tensors += 1;
                *num_tensors - 1
            });

            nodes[i].inputs.push(*en);
        }

        for out in &n.output {
            let en = tensor_mapping.entry(out.to_string()).or_insert_with(|| {
                *num_tensors += 1;
                *num_tensors - 1
            });

            nodes[i].outputs.push(*en);
        }
    }
}

/**
 * Generate tensor -> buffer mappings given input nodes
 *
 * @param nodes Nodes in the logical graph
 * @param buffers Mutable reference to tensor -> buffer mapping to be filled. buf_idx := buffer[tens_idx].
 * @param tensor_size Number of intermediate tensors
 * @param tensor_mapping Mapping from tensor name -> tensor idx.
 *
 * @return Size of the (locally) minimum number of buffers required
 */
fn generate_buffer_mappings(nodes: &[Node], buffers: &mut [usize], tensor_size: usize) -> usize {
    let mut adjacencies = vec![Vec::<usize>::new(); tensor_size];

    // Should track the start node this tensor data is required, and end node
    let mut liveness: Vec<(usize, usize)> = vec![(0, nodes.len() + 1); tensor_size];

    // We need to map tensors to buffers, by first generating tensor liveness.
    for (i, node) in nodes.iter().enumerate() {
        for &inp in &node.inputs {
            liveness[inp].1 = i; // Must be consumed earliest this node.
        }

        for &out in &node.outputs {
            liveness[out].0 = i; // Must be created latest this node.
        }
    }

    // Iterate through all possible liveness pairs
    for i in 0..tensor_size {
        for j in i + 1..tensor_size {
            let (a, b) = liveness[i];
            let (p, q) = liveness[j];

            if p <= b && q >= a {
                adjacencies[i].push(j);
                adjacencies[j].push(i);
            }
        }
    }

    let mut max_buf = 0;
    let mut visited = vec![false; tensor_size];

    for tens_idx in 0..tensor_size {
        visited[tens_idx] = true;

        let mut adj_bufs = HashSet::with_capacity(adjacencies[tens_idx].len());

        for &nbor in &adjacencies[tens_idx] {
            if !visited[nbor] || nbor == tens_idx {
                continue;
            }

            adj_bufs.insert(buffers[nbor]);
        }

        let mut buf_idx = 0;
        while adj_bufs.contains(&buf_idx) {
            buf_idx += 1;
        }

        buffers[tens_idx] = buf_idx;
        max_buf = max_buf.max(buf_idx);
    }

    max_buf + 1
}

fn parse_op(node: &NodeProto, initializers: &HashMap<String, Tensor>) -> Result<ops::Op, String> {
    let attrs = parse_attrs(node.attribute.as_slice());

    let op = match node.op_type.as_str() {
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
            let output_shape: TensorShape = to_stack_shape(
                shape_tensor
                    .data::<i64>()?
                    .iter()
                    .map(|&v| v as usize)
                    .collect(),
            )?;

            ops::Op::Reshape(ops::ReshapeData { output_shape })
        }
        op => todo!("{} op not supported", op),
    };

    Ok(op)
}

fn parse_input_shapes(
    shapes: &mut [TensorShape],
    types: &mut [TensorType],
    inputs: &[ValueInfoProto],
    initializers: &HashMap<String, Tensor>,
    tensor_mapping: &HashMap<String, usize>,
) -> Result<(), String> {
    for inp in inputs {
        let input_type = inp
            .r#type
            .as_ref()
            .ok_or_else(|| format!("Expected graph input {} to have a type", inp.name))?;

        let tens_idx = *tensor_mapping.get(&inp.name).ok_or_else(|| format!(""))?;

        match &input_type.value {
            Some(proto::type_proto::Value::TensorType(tensor)) => {
                use proto::tensor_shape_proto::dimension::Value::DimValue;
                let shape_proto = tensor
                    .shape
                    .as_ref()
                    .ok_or_else(|| "Tensor shape expected.")?;

                use proto::tensor_proto::DataType;

                types[tens_idx] = match DataType::try_from(tensor.elem_type)
                    .map_err(|d| format!("Invalid tensor element type encountered {}", d))?
                {
                    DataType::Float => TensorType::F32,
                    DataType::Int64 => TensorType::I64,
                    _ => panic!(),
                };

                shapes[tens_idx] = to_stack_shape(
                    shape_proto
                        .dim
                        .iter()
                        .map(|d| match d.value {
                            Some(DimValue(v)) => v as usize,
                            _ => panic!(),
                        })
                        .collect(),
                )?; // TODO(satvik): Allow to_stack_shape to take in an iterator
            }
            _ => todo!("{:#?} not supported", input_type.value),
        }
    }

    // Parse initializer shapes
    for (tensor_name, tensor) in initializers {
        let tens_idx = *tensor_mapping.get(tensor_name).ok_or_else(|| format!(""))?;

        shapes[tens_idx] = tensor.shape();
        types[tens_idx] = tensor.get_type();
    }

    Ok(())
}

fn to_stack_shape(shape: Vec<usize>) -> Result<TensorShape, String> {
    if shape.len() > MAX_RANK {
        return Err(format!(
            "tinf only supports tensors with max rank {}, encountered rank {}",
            MAX_RANK,
            shape.len()
        ));
    }

    let mut stack_shape = [1usize; MAX_RANK];

    for (i, &dim) in shape.iter().rev().enumerate() {
        stack_shape[MAX_RANK - 1 - i] = dim;
    }

    Ok(stack_shape)
}

/**
 * Parses initializer tensor from a TensorProto. Performs no allocations for tensor data.
 *
 * @param t TensorProto to parse
 * @return (name, tensor)
 */
fn parse_initializer<'a>(t: &'a mut proto::TensorProto) -> Result<(String, Tensor<'a>), String> {
    use proto::tensor_proto::DataType;
    use std::convert::TryFrom;

    let dt = DataType::try_from(t.data_type)
        .map_err(|_| format!("Unknown datatype encountered: {}", t.data_type))?;
    let shape: TensorShape = to_stack_shape(t.dims.iter().map(|&d| d as usize).collect())?;

    match dt {
        DataType::Float => Ok((t.name.clone(), Tensor::new(&mut t.float_data, shape))),
        DataType::Int64 => Ok((t.name.clone(), Tensor::new(&mut t.int64_data, shape))),
        _ => {
            println!("{:#?}", dt);
            todo!();
        }
    }
}

pub fn parse(path: &str) -> Result<Model, String> {
    let bytes = fs::read(path).map_err(|e| format!("Error reading file {}", e))?;
    let proto = proto::ModelProto::decode(bytes.as_slice())
        .map_err(|e| format!("Error decoding ONNX format: {}", e))?;

    let mut g = proto
        .graph
        .ok_or_else(|| "Model did not contain a graph.")?;

    let mut tensor_size = 0usize;
    let mut tensor_mapping =
        HashMap::<String, usize>::with_capacity(g.node.len() + g.input.len() + g.output.len());

    let initializers: HashMap<String, Tensor> = g
        .initializer
        .iter_mut()
        .map(|init| parse_initializer(init))
        .collect::<Result<_, _>>()?;

    let mut nodes: Vec<Node> = g
        .node
        .iter()
        .map(|n| Node {
            inputs: Vec::with_capacity(n.input.len()),
            outputs: Vec::with_capacity(n.output.len()),
            op: parse_op(n, &initializers).unwrap(), // TODO(satvik): Parse the operation here properly
        })
        .collect(); // Pre-allocate space for nodes.

    parse_nodes_and_generate_tensor_mappings(
        &g.node,
        &mut nodes,
        &mut tensor_mapping,
        &mut tensor_size,
    );

    // Maps tensors to their corresponding buffers
    let mut buffers = vec![0usize; tensor_size];
    let num_buffers =
        generate_buffer_mappings(nodes.as_slice(), buffers.as_mut_slice(), tensor_size);

    let mut shapes = vec![[0usize; MAX_RANK]; tensor_size]; // Shapes for each in/out tensor
    let mut types = vec![TensorType::F32; tensor_size];

    // Shape inference
    {
        parse_input_shapes(
            &mut shapes.as_mut_slice(),
            &mut types.as_mut_slice(),
            &g.input,
            &initializers,
            &tensor_mapping,
        )?;

        for node in &nodes {
            parse_tensor_shapes(node, shapes.as_mut_slice(), types.as_mut_slice());
        }
    }

    // For each buffer, compute the largest tensor stored in that buffer
    let mut buffer_sizes = vec![0usize; num_buffers];
    for (tensor, &buffer) in buffers.iter().enumerate() {
        let num_elements = shapes[tensor].iter().product::<usize>();
        buffer_sizes[buffer] = buffer_sizes[buffer].max(num_elements * types[tensor].elem_size());
    }

    let mut offset = 0;
    let buffer_offsets: Vec<usize> = buffer_sizes
        .iter()
        .map(|sz| {
            offset += sz;
            offset - sz
        })
        .collect();

    // 64 byte alignment for a cache line.
    let layout = Layout::from_size_align(buffer_sizes.iter().sum::<usize>(), 64)
        .map_err(|e| format!("Error allocating arena: {}", e))?;
    let arena = unsafe { alloc_zeroed(layout) };

    let init_indexed = initializers
        .into_iter()
        .map(|(k, v)| (tensor_mapping[&k], Vec::from(&v)))
        .collect();

    Ok(Model {
        nodes,
        tensor_buffer_mapping: buffers,
        tensor_shapes: shapes,
        tensor_types: types,

        buffer_sizes: buffer_sizes,
        buffer_offsets: buffer_offsets,

        arena_layout: layout,
        arena,

        initializers: init_indexed,
    })
}
