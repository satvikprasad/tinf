use crate::{
    model::Model,
    tensor::{self, MAX_RANK, Tensor, TensorShape, TensorView},
};

pub trait InferShape {
    fn infer_shape(&self, inputs: &[TensorShape], output: &mut TensorShape); // TODO(satvik): Maybe make user allocate this vector
}

pub trait Execute {
    fn execute(&self, inputs: &[usize], outputs: &[usize], model: &Model);
}

#[derive(Debug)]
pub enum AutoPadType {
    SameUpper,
    SameLower,
    NotSet,
}

#[derive(Debug)]
pub struct ConvData {
    pub pads: [usize; 4],
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub group: usize,
    pub pad_type: AutoPadType,
}

impl InferShape for ConvData {
    fn infer_shape(&self, inputs: &[TensorShape], output: &mut TensorShape) {
        let input_dim = inputs[0]; // [N, C, H, W]
        let kernel_dim = inputs[1]; // [O, I, H, W]

        debug_assert_eq!(
            input_dim[1],
            kernel_dim[1] * self.group,
            "Input channels must match kernel channels * groups"
        );

        match self.pad_type {
            AutoPadType::NotSet => {
                let effective_dim = [
                    self.dilations[0] * (kernel_dim[2] - 1) + 1,
                    self.dilations[1] * (kernel_dim[3] - 1) + 1,
                ];

                let out_h = (input_dim[2] + self.pads[0] + self.pads[2] - effective_dim[0])
                    / self.strides[0]
                    + 1;
                let out_w = (input_dim[3] + self.pads[1] + self.pads[3] - effective_dim[1])
                    / self.strides[1]
                    + 1;

                output[0] = input_dim[0];
                output[1] = kernel_dim[0];
                output[2] = out_h;
                output[3] = out_w;
            }

            _ => {
                output[0] = input_dim[0];
                output[1] = kernel_dim[0];
                output[2] = input_dim[2].div_ceil(self.strides[0]);
                output[3] = input_dim[3].div_ceil(self.strides[1]);
            }
        }
    }
}

impl ConvData {
    pub fn compute_autopad(&self, inputs: &[TensorShape], out_pads: &mut [usize; 4]) {
        // TODO(satvik): Make this branchless.

        let Self {
            pads: _,
            strides,
            dilations,
            group: _,
            pad_type,
        } = self;

        assert!(
            !matches!(pad_type, AutoPadType::NotSet),
            "Cannot compute autopad when autopad is not set."
        );

        let input_shape = inputs[0];
        let kernel_shape = inputs[1];

        let effective_dim = [
            dilations[0] * (kernel_shape[2] - 1) + 1,
            dilations[1] * (kernel_shape[3] - 1) + 1,
        ];

        let ypads = strides[0] * (input_shape[2].div_ceil(strides[0]) - 1) + effective_dim[0]
            - input_shape[2];
        let xpads = strides[1] * (input_shape[3].div_ceil(strides[1]) - 1) + effective_dim[1]
            - input_shape[3];

        if xpads % 2 == 0 {
            out_pads[1] = xpads / 2;
            out_pads[3] = xpads / 2;
        } else {
            match pad_type {
                AutoPadType::SameLower => {
                    out_pads[1] = xpads / 2 + 1;
                    out_pads[3] = xpads / 2;
                }

                AutoPadType::SameUpper => {
                    out_pads[1] = xpads / 2;
                    out_pads[3] = xpads / 2 + 1;
                }

                _ => panic!(),
            }
        }

        if ypads % 2 == 0 {
            out_pads[0] = ypads / 2;
            out_pads[2] = ypads / 2;
        } else {
            match pad_type {
                AutoPadType::SameLower => {
                    out_pads[0] = ypads / 2 + 1;
                    out_pads[2] = ypads / 2;
                }

                AutoPadType::SameUpper => {
                    out_pads[0] = ypads / 2;
                    out_pads[2] = ypads / 2 + 1;
                }

                _ => panic!(),
            }
        }
    }
}

#[derive(Debug)]
pub struct MaxPoolData {
    pub pads: [usize; 4],
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub kernel_shape: [usize; 2],
    pub pad_type: AutoPadType,
}

impl InferShape for MaxPoolData {
    fn infer_shape(&self, inputs: &[TensorShape], output: &mut TensorShape) {
        let input_dim = inputs[0]; // [N, C, H, W]

        match self.pad_type {
            AutoPadType::NotSet => {
                let effective_dim = [
                    self.dilations[0] * (self.kernel_shape[0] - 1) + 1,
                    self.dilations[1] * (self.kernel_shape[1] - 1) + 1,
                ];

                let out_h = (input_dim[2] + self.pads[0] + self.pads[2] - effective_dim[0])
                    / self.strides[0]
                    + 1;
                let out_w = (input_dim[3] + self.pads[1] + self.pads[3] - effective_dim[1])
                    / self.strides[1]
                    + 1;

                output[0] = input_dim[0];
                output[1] = input_dim[1];
                output[2] = out_h;
                output[3] = out_w;
            }
            _ => {
                output[0] = input_dim[0];
                output[1] = input_dim[1];
                output[2] = input_dim[2].div_ceil(self.strides[0]);
                output[3] = input_dim[3].div_ceil(self.strides[1]);
            }
        }
    }
}

impl MaxPoolData {
    pub fn compute_autopad(&self, inputs: &[TensorShape], out_pads: &mut [usize; 4]) {
        // TODO(satvik): Make this branchless.

        let Self {
            pads,
            strides,
            dilations,
            pad_type,
            kernel_shape,
        } = self;

        if let AutoPadType::NotSet = pad_type {
            out_pads.copy_from_slice(pads);
            return;
        }

        let input_shape = inputs[0];

        let effective_dim = [
            dilations[0] * (kernel_shape[0] - 1) + 1,
            dilations[1] * (kernel_shape[1] - 1) + 1,
        ];

        let ypads = strides[0] * (input_shape[2].div_ceil(strides[0]) - 1) + effective_dim[0]
            - input_shape[2];
        let xpads = strides[1] * (input_shape[3].div_ceil(strides[1]) - 1) + effective_dim[1]
            - input_shape[3];

        if xpads % 2 == 0 {
            out_pads[1] = xpads / 2;
            out_pads[3] = xpads / 2;
        } else {
            match pad_type {
                AutoPadType::SameLower => {
                    out_pads[1] = xpads / 2 + 1;
                    out_pads[3] = xpads / 2;
                }

                AutoPadType::SameUpper => {
                    out_pads[1] = xpads / 2;
                    out_pads[3] = xpads / 2 + 1;
                }

                _ => panic!(),
            }
        }

        if ypads % 2 == 0 {
            out_pads[0] = ypads / 2;
            out_pads[2] = ypads / 2;
        } else {
            match pad_type {
                AutoPadType::SameLower => {
                    out_pads[0] = ypads / 2 + 1;
                    out_pads[2] = ypads / 2;
                }

                AutoPadType::SameUpper => {
                    out_pads[0] = ypads / 2;
                    out_pads[2] = ypads / 2 + 1;
                }

                _ => panic!(),
            }
        }
    }
}

#[derive(Debug)]
pub struct ReshapeData {
    pub output_shape: TensorShape,
}

impl InferShape for ReshapeData {
    fn infer_shape(&self, _: &[TensorShape], output: &mut TensorShape) {
        output.clone_from(&self.output_shape);
    }
}

#[derive(Debug)]
pub enum Op {
    Conv(ConvData),
    Add,
    MaxPool(MaxPoolData),
    MatMul,
    Reshape(ReshapeData),
    ReLU,
}

impl InferShape for Op {
    fn infer_shape(&self, inputs: &[TensorShape], output: &mut TensorShape) {
        match self {
            Op::Conv(cd) => cd.infer_shape(inputs, output),
            Op::MaxPool(mpd) => mpd.infer_shape(inputs, output),
            Op::Add => {
                // TODO(satvik): Broadcasting validation.
                use crate::tensor;

                let a = inputs[0];
                let b = inputs[1];

                for i in 0..tensor::MAX_RANK {
                    output[i] = a[i].max(b[i]);
                }
            }
            Op::MatMul => {
                // TODO(satvik): Broadcasting validation.
                use crate::tensor;

                let a = inputs[0];
                let b = inputs[1];

                assert_eq!(a[tensor::MAX_RANK - 1], b[tensor::MAX_RANK - 2]);

                output[tensor::MAX_RANK - 1] = b[tensor::MAX_RANK - 1];
                output[tensor::MAX_RANK - 2] = a[tensor::MAX_RANK - 2];

                for i in 0..tensor::MAX_RANK - 2 {
                    output[i] = a[i].max(b[i]);
                }
            }
            Op::Reshape(rd) => rd.infer_shape(inputs, output),
            Op::ReLU => {
                output.clone_from(&inputs[0]);
            }
        };
    }
}

macro_rules! retr_tensor {
    ($model:ident, $tens:expr) => {
        unsafe {
            Tensor::from_arena(
                $model.arena,
                $model.buffer_offsets[$model.tensor_buffer_mapping[$tens]],
                $model.tensor_shapes[$tens],
                $model.tensor_types[$tens],
            )
        }
    };
}

impl Execute for Op {
    fn execute(&self, inputs: &[usize], outputs: &[usize], model: &Model) {
        match self {
            Op::ReLU => {
                let in_tensor = retr_tensor!(model, inputs[0]);
                let mut out_tensor = retr_tensor!(model, outputs[0]);

                for (out_elem, in_elem) in out_tensor
                    .data_mut_unchecked::<f32>()
                    .iter_mut()
                    .zip(in_tensor.data_unchecked::<f32>().iter())
                {
                    *out_elem = in_elem.max(0f32);
                }
            }
            Op::Add => {
                let a = retr_tensor!(model, inputs[0]);
                let a_data = a.data_unchecked::<f32>();

                let b = retr_tensor!(model, inputs[1]);
                let b_data = b.data_unchecked::<f32>();

                let mut out_tensor = retr_tensor!(model, outputs[0]);

                let out_size = out_tensor.shape().iter().product::<usize>();
                let out_strides = out_tensor.strides();
                let out_tensor_data = out_tensor.data_mut_unchecked::<f32>();

                for i in 0..out_size {
                    let mut curr = i;

                    let mut a_idx = 0;
                    let mut b_idx = 0;

                    for k in 0..MAX_RANK {
                        let index = curr / out_strides[k];

                        a_idx += index.min(a.shape()[k] - 1) * a.strides()[k];
                        b_idx += index.min(b.shape()[k] - 1) * b.strides()[k];

                        curr = curr % out_strides[k];
                    }

                    out_tensor_data[i] = a_data[a_idx] + b_data[b_idx];
                }
            }
            Op::MatMul => {
                let a = retr_tensor!(model, inputs[0]);
                let a_data: &[f32] = a.data_unchecked();

                let b = retr_tensor!(model, inputs[1]);
                let b_data: &[f32] = b.data_unchecked();

                let mut out_tensor = retr_tensor!(model, outputs[0]);
                let out_size = out_tensor.shape().iter().product::<usize>();
                let out_strides = out_tensor.strides();
                let out_tensor_data: &mut [f32] = out_tensor.data_mut_unchecked();

                // TODO(satvik): Transpose b_data for better cache locality.
                for i in 0..out_size {
                    let mut curr = i;

                    let mut a_start = 0;
                    let mut b_start = 0;

                    for k in 0..MAX_RANK - 2 {
                        a_start += (curr / out_strides[k]).min(a.shape()[k] - 1) * a.strides()[k];
                        b_start += (curr / out_strides[k]).min(b.shape()[k] - 1) * b.strides()[k];

                        curr = curr % out_strides[k];
                    }

                    out_tensor_data[i] = 0f32;

                    let m = curr / out_strides[MAX_RANK - 2];
                    let n = (curr % out_strides[MAX_RANK - 2]) / out_strides[MAX_RANK - 1];

                    for k in 0..a.shape()[MAX_RANK - 1] {
                        out_tensor_data[i] += a_data[a_start + a.strides()[MAX_RANK - 2] * m + k]
                            * b_data[b_start + k * b.strides()[MAX_RANK - 2] + n];
                    }
                }
            }
            Op::Conv(conv_data) => {
                if conv_data.group != 1 {
                    todo!();
                }

                let input = retr_tensor!(model, inputs[0]);
                let input_data: &[f32] = input.data_unchecked();
                let input_strides = input.strides();
                let input_shape = input.shape();

                let kernel = retr_tensor!(model, inputs[1]);
                let kernel_data: &[f32] = kernel.data_unchecked();
                let kernel_strides = kernel.strides();
                let kernel_shape = kernel.shape();

                let mut output = retr_tensor!(model, outputs[0]);
                let out_strides = output.strides();
                let output_shape = output.shape();

                let output_data: &mut [f32] = output.data_mut_unchecked();

                let mut out_pads = [0usize; 4];
                conv_data.compute_autopad(&[input.shape(), kernel.shape()], &mut out_pads);

                // TODO(satvik): Support other dimension convolutions.
                for n in 0..output_shape[0] {
                    for out_ch in 0..output_shape[1] {
                        for out_row in 0..output_shape[2] {
                            for out_col in 0..output_shape[3] {
                                let mut sum = 0.0;

                                for in_ch in 0..input_shape[1] {
                                    for k_row in 0..kernel_shape[2] {
                                        let input_row: isize = (out_row * conv_data.strides[0]
                                            + k_row * conv_data.dilations[0])
                                            as isize
                                            - out_pads[0] as isize;

                                        if (input_row as usize) >= input_shape[2] {
                                            continue;
                                        }

                                        for k_col in 0..kernel_shape[3] {
                                            let input_col: isize = (out_col * conv_data.strides[1]
                                                + k_col * conv_data.dilations[1])
                                                as isize
                                                - out_pads[1] as isize;

                                            if (input_col as usize) >= input_shape[3] {
                                                continue;
                                            }

                                            let in_idx = n * input_strides[0]
                                                + in_ch * input_strides[1]
                                                + (input_row as usize) * input_strides[2]
                                                + (input_col as usize) * input_strides[3];

                                            let k_idx = out_ch * kernel_strides[0]
                                                + in_ch * kernel_strides[1]
                                                + k_row * kernel_strides[2]
                                                + k_col * kernel_strides[3];

                                            sum += input_data[in_idx] * kernel_data[k_idx];
                                        }
                                    }
                                }

                                let out_idx = n * out_strides[0]
                                    + out_ch * out_strides[1]
                                    + out_row * out_strides[2]
                                    + out_col * out_strides[3];

                                output_data[out_idx] = sum;
                            }
                        }
                    }
                }
            }
            Op::MaxPool(max_pool_data) => {
                let input = retr_tensor!(model, inputs[0]);
                let input_data: &[f32] = input.data_unchecked();
                let input_strides = input.strides();
                let input_shape = input.shape();

                let mut output = retr_tensor!(model, outputs[0]);
                let out_strides = output.strides();
                let output_shape = output.shape();

                let output_data: &mut [f32] = output.data_mut_unchecked();

                let mut out_pads = [0usize; 4];
                max_pool_data.compute_autopad(&[input.shape()], &mut out_pads);

                // TODO(satvik): Support other dimension convolutions.
                for n in 0..output_shape[0] {
                    for ch in 0..output_shape[1] {
                        for out_row in 0..output_shape[2] {
                            for out_col in 0..output_shape[3] {
                                let mut max_entry = f32::NEG_INFINITY;

                                for k_row in 0..max_pool_data.kernel_shape[0] {
                                    let input_row: isize = (out_row * max_pool_data.strides[0]
                                        + k_row * max_pool_data.dilations[0])
                                        as isize
                                        - out_pads[0] as isize;

                                    if (input_row as usize) >= input_shape[2] {
                                        continue;
                                    }

                                    for k_col in 0..max_pool_data.kernel_shape[1] {
                                        let input_col: isize = (out_col * max_pool_data.strides[1]
                                            + k_col * max_pool_data.dilations[1])
                                            as isize
                                            - out_pads[1] as isize;

                                        if (input_col as usize) >= input_shape[3] {
                                            continue;
                                        }

                                        let in_idx = n * input_strides[0]
                                            + ch * input_strides[1]
                                            + (input_row as usize) * input_strides[2]
                                            + (input_col as usize) * input_strides[3];

                                        max_entry = max_entry.max(input_data[in_idx]);
                                    }
                                }

                                let out_idx = n * out_strides[0]
                                    + ch * out_strides[1]
                                    + out_row * out_strides[2]
                                    + out_col * out_strides[3];

                                output_data[out_idx] = max_entry;
                            }
                        }
                    }
                }
            }
            Op::Reshape(_) => {
                let a = retr_tensor!(model, inputs[0]);
                let a_data: &[f32] = a.data_unchecked();

                let out_size = a.shape().iter().product();

                let mut output = retr_tensor!(model, outputs[0]);
                let output_data: &mut [f32] = output.data_mut_unchecked();

                for i in 0..out_size {
                    output_data[i] = a_data[i];
                }
            }
        }
    }
}
