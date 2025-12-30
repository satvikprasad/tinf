use std::ops::Add;

#[derive(Clone, Debug)]
pub struct TensorData<T: Copy + Default + Add<Output = T>> {
    data: Vec<T>,
    strides: Vec<usize>,

    pub shape: Vec<usize>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

impl<T: Copy + Default + Add<Output = T>> TensorData<T> {
    pub fn index(&self, coords: &[usize]) -> usize {
        coords.iter().zip(&self.strides).map(|(c, s)| c * s).sum()
    }

    pub fn get(&self, coords: &[usize]) -> T {
        self.data[self.index(coords)]
    }

    pub fn set(&mut self, coords: &[usize], val: T) {
        let idx = self.index(coords);
        self.data[idx] = val;
    }

    pub fn add(&self, other: &TensorData<T>, output: &mut TensorData<T>) {
        // Ensure dimensions are accurate
        assert_eq!(self.shape, other.shape);
        assert_eq!(other.shape, output.shape);

        for ((a, b), c) in output.data.iter_mut().zip(&self.data).zip(&other.data) {
            *a = *b + *c;
        }
    }
}

#[derive(Debug)]
pub enum Tensor {
    F32(TensorData<f32>),
    I64(TensorData<i64>),
}

impl Tensor {
    pub fn new<T: IntoTensor>(data: Vec<T>, shape: Vec<usize>) -> Self {
        // Assert that data is large enough
        T::into_tensor(data, shape)
    }

    pub fn data<T: IntoTensor>(&self) -> Option<&[T]> {
        T::try_get_data(self)
    }

    pub fn splat<T: IntoTensor + Clone>(val: T, shape: Vec<usize>) -> Self {
        let data = vec![val; shape.iter().product()];
        Self::new(data, shape)
    }

    pub fn zeros<T: IntoTensor + Clone + Default>(shape: Vec<usize>) -> Self {
        Self::splat::<T>(Default::default(), shape)
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(t) => &t.shape,
            Tensor::I64(t) => &t.shape,
        }
    }
}

pub trait IntoTensor {
    fn into_tensor(data: Vec<Self>, shape: Vec<usize>) -> Tensor
    where
        Self: Sized;

    fn try_get_data(tens: &Tensor) -> Option<&[Self]>
    where
        Self: Sized;
}

macro_rules! impl_into_tensor {
    ($t:ty, $variant:ident) => {
        impl IntoTensor for $t {
            fn into_tensor(data: Vec<Self>, shape: Vec<usize>) -> Tensor {
                let strides = compute_strides(&shape);
                Tensor::$variant(TensorData {
                    data,
                    shape,
                    strides,
                })
            }

            fn try_get_data(tens: &Tensor) -> Option<&[Self]> {
                match tens {
                    Tensor::$variant(t) => Some(t.data.as_slice()),
                    _ => None,
                }
            }
        }
    };
}

impl_into_tensor!(f32, F32);
impl_into_tensor!(i64, I64);
