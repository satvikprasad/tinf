use std::alloc::Layout;

use crate::{ops, tensor::{Tensor, TensorShape}};

#[derive(Debug)]
pub struct Node {
    pub op: ops::Op,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Debug)]
pub struct Model {
    /** Topologically sorted series of nodes. */
    pub nodes: Vec<Node>,
    pub tensor_buffer_mapping: Vec<usize>,
    pub tensor_shapes: Vec<(TensorShape, usize)>,

    pub arena: *mut u8,
    pub arena_layout: Layout,

    pub buffer_offsets: Vec<usize>,
    pub buffer_sizes: Vec<usize>, // TODO(satvik): Perhaps store raw pointers instead if we're in a hot
                                  // loop.
}

impl Node {
    pub fn execute(&self) {}
}

impl Model {
    pub fn execute(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!("Executing node {}...", i);
        }
    }
}
