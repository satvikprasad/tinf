use crate::tensor::Tensor;

pub trait InferShape {
    fn infer_shape(&self, inputs: &[&[usize]], output: &mut Vec<usize>); // TODO(satvik): Maybe make user allocate this vector
}

pub trait Execute {
    fn execute(&self, inputs: &[&Tensor], output: &mut Tensor);
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
    fn infer_shape(&self, inputs: &[&[usize]], output: &mut Vec<usize>) {
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

                output.resize(4, 0);

                output[0] = input_dim[0];
                output[1] = kernel_dim[0];
                output[2] = out_h;
                output[3] = out_w;
            }

            _ => {
                output.resize(4, 0);

                output[0] = input_dim[0];
                output[1] = kernel_dim[0];
                output[2] = input_dim[2].div_ceil(self.strides[0]);
                output[3] = input_dim[3].div_ceil(self.strides[1]);
            }
        }
    }
}

impl ConvData {
    pub fn compute_autopad(
        inputs: &[&[usize]],
        dilations: &[usize; 2],
        strides: &[usize; 2],
        out_pads: &mut [usize; 4],
        pad_type: &AutoPadType,
    ) {
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
    fn infer_shape(&self, inputs: &[&[usize]], output: &mut Vec<usize>) {
        output.resize(4, 0);
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

#[derive(Debug)]
pub struct ReshapeData {
    pub output_shape: Vec<usize>,
}

impl InferShape for ReshapeData {
    fn infer_shape(&self, _: &[&[usize]], output: &mut Vec<usize>) {
        output.resize(self.output_shape.len(), 0);

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
    fn infer_shape(&self, inputs: &[&[usize]], output: &mut Vec<usize>) {
        match self {
            Op::Conv(cd) => cd.infer_shape(inputs, output),
            Op::MaxPool(mpd) => mpd.infer_shape(inputs, output),
            Op::Add => {
                let a = &inputs[0];
                let b = &inputs[1];

                let max_rank = a.len().max(b.len());

                output.resize(max_rank, 0);

                let mut i = a.len() as i32 - 1;
                let mut j = b.len() as i32 - 1;
                let mut k = max_rank as i32 - 1;

                while i >= 0 || j >= 0 {
                    if i < 0 {
                        output[k as usize] = b[j as usize];
                        j -= 1;
                        k -= 1;
                        continue;
                    }

                    if j < 0 {
                        output[k as usize] = a[i as usize];
                        i -= 1;
                        k -= 1;
                        continue;
                    }

                    output[k as usize] = a[i as usize].max(b[j as usize]);

                    i -= 1;
                    j -= 1;
                    k -= 1;
                }
            }
            Op::MatMul => {
                let a = &inputs[0];
                let b = &inputs[1];

                let max_rank = a.len().max(b.len());
                output.resize(max_rank, 0);

                assert_eq!(a[a.len() - 1], b[b.len() - 2]);

                if a.len() < 2 || b.len() < 2 {
                    todo!("1D matmul not implemented yet.");
                }

                output[max_rank - 1] = b[b.len() - 1];
                output[max_rank - 2] = a[a.len() - 2];

                let mut i = a.len() as i32 - 3;
                let mut j = b.len() as i32 - 3;
                let mut k = max_rank as i32 - 3;

                while i >= 0 || j >= 0 {
                    if i < 0 {
                        output[k as usize] = b[j as usize];
                        j -= 1;
                        k -= 1;
                        continue;
                    }

                    if j < 0 {
                        output[k as usize] = a[i as usize];
                        i -= 1;
                        k -= 1;
                        continue;
                    }

                    output[k as usize] = a[i as usize].max(b[j as usize]);

                    i -= 1;
                    j -= 1;
                    k -= 1;
                }
            }
            Op::Reshape(rd) => {
                rd.infer_shape(inputs, output);
            }
            Op::ReLU => {
                output.resize(inputs[0].len(), 0);

                output.clone_from_slice(inputs[0]);
            }
        };
    }
}
