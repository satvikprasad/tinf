use crate::tensor::Tensor;

pub trait InferShape {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>; // TODO(satvik): Maybe make user allocate this vector
}

pub trait Execute {
    fn execute(&self, inputs: &[&Tensor], output: &mut Tensor);
}

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
}

impl InferShape for ConvData {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_dim = inputs[0]; // [N, C, H, W]
        let kernel_dim = inputs[1]; // [O, I, H, W]
        //
        debug_assert_eq!(
            input_dim[1],
            kernel_dim[1] * self.group,
            "Input channels must match kernel channels * groups"
        );

        let effective_dim = [
            self.dilations[0] * (kernel_dim[2] - 1) + 1,
            self.dilations[1] * (kernel_dim[3] - 1) + 1,
        ];

        let out_h =
            (input_dim[2] + self.pads[0] + self.pads[2] - effective_dim[0]) / self.strides[0] + 1;
        let out_w =
            (input_dim[3] + self.pads[1] + self.pads[3] - effective_dim[1]) / self.strides[1] + 1;

        vec![input_dim[0], kernel_dim[0], out_h, out_w]
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
    pub kernel_shape: [usize; 2]
}

impl InferShape for MaxPoolData {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_dim = inputs[0]; // [N, C, H, W]

        let effective_dim = [
            self.dilations[0] * (self.kernel_shape[0] - 1) + 1,
            self.dilations[1] * (self.kernel_shape[1] - 1) + 1,
        ];

        let out_h =
            (input_dim[2] + self.pads[0] + self.pads[2] - effective_dim[0]) / self.strides[0] + 1;
        let out_w =
            (input_dim[3] + self.pads[1] + self.pads[3] - effective_dim[1]) / self.strides[1] + 1;

        vec![input_dim[0], input_dim[1], out_h, out_w]
    }
}

#[derive(Debug)]
pub enum Op {
    Conv(ConvData),
    Add,
    MaxPool(MaxPoolData),
}

impl InferShape for Op {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        match self {
            Op::Conv(cd) => cd.infer_shape(inputs),
            Op::MaxPool(mpd) => mpd.infer_shape(inputs),
            Op::Add => {
                let a = &inputs[0];
                let b = &inputs[1];

                let max_rank = a.len().max(b.len());
                let mut shape = vec![0usize; max_rank];

                let mut i = (a.len() - 1) as i32;
                let mut j = (b.len() - 1) as i32;
                let mut k = (max_rank - 1) as i32;

                while i >= 0 || j >= 0 {
                    if i < 0 {
                        shape[k as usize] = b[j as usize];
                        j -= 1;
                        k -= 1;
                        continue;
                    }

                    if j < 0 {
                        shape[k as usize] = a[i as usize];
                        i -= 1;
                        k -= 1;
                        continue;
                    }

                    shape[k as usize] = a[i as usize].max(b[j as usize]);

                    i -= 1;
                    j -= 1;
                    k -= 1;
                }

                shape
            }
        }
    }
}
