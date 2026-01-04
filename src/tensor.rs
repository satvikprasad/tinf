pub const MAX_RANK: usize = 4;

pub type TensorShape = [usize; MAX_RANK];

#[derive(Debug)]
pub struct TensorView<'a, T> {
    pub data: &'a mut [T],
    pub shape: TensorShape, // Capped on max_rank = 4
}

impl<'a, T> TensorView<'a, T> {
    pub fn from_buffer(data: &'a mut [T], shape: TensorShape) -> Self {
        Self { data, shape }
    }
}

#[derive(Debug)]
pub enum Tensor<'a> {
    F32(TensorView<'a, f32>),
    F64(TensorView<'a, f64>),
    I64(TensorView<'a, i64>),
}

pub trait ToTensor {
    fn as_tensor<'a>(data: &'a mut Vec<Self>, shape: TensorShape) -> Tensor<'a>
    where
        Self: Sized;

    fn as_tensor_sliced<'a>(data: &'a mut [Self], shape: TensorShape) -> Tensor<'a>
    where
        Self: Sized;

    fn from_tensor<'a>(tensor: &'a Tensor) -> Result<&'a [Self], String>
    where
        Self: Sized;
}

macro_rules! impl_to_tensor {
    ($t:ty, $variant:ident) => {
        impl ToTensor for $t {
            fn as_tensor<'a>(data: &'a mut Vec<Self>, shape: TensorShape) -> Tensor<'a>
            where
                Self: Sized,
            {
                return Tensor::$variant(TensorView {
                    data: data.as_mut_slice(),
                    shape,
                });
            }

            fn from_tensor<'a>(tensor: &'a Tensor) -> Result<&'a [Self], String>
            where
                Self: Sized,
            {
                match tensor {
                    Tensor::$variant(view) => Ok(view.data),
                    _ => Err(format!("Invalid tensor type retrieval")),
                }
            }

            fn as_tensor_sliced<'a>(data: &'a mut [Self], shape: TensorShape) -> Tensor<'a>
            where
                Self: Sized,
            {
                return Tensor::$variant(TensorView { data, shape });
            }
        }
    };
}

impl_to_tensor!(f32, F32);
impl_to_tensor!(f64, F64);
impl_to_tensor!(i64, I64);

impl<'a> Tensor<'a> {
    pub fn new<T: ToTensor>(data: &'a mut Vec<T>, shape: TensorShape) -> Self {
        return T::as_tensor(data, shape);
    }

    pub unsafe fn from_arena<T: ToTensor + 'a>(arena: *mut u8, byte_offset: usize, shape: TensorShape) -> Self {
        let data = unsafe {
            std::slice::from_raw_parts_mut(arena.add(byte_offset) as *mut T, shape.iter().product())
        };

        T::as_tensor_sliced(data, shape)
    }

    pub fn shape(&self) -> TensorShape {
        match self {
            Self::F32(view) => view.shape,
            Self::I64(view) => view.shape,
            Self::F64(view) => view.shape,
        }
    }

    pub fn data<T: ToTensor>(&'a self) -> Result<&'a [T], String> {
        T::from_tensor(self)
    }

    pub fn elem_size(&self) -> usize {
        match self {
            Self::F32(_) => size_of::<f32>(),
            Self::F64(_) => size_of::<f64>(),
            Self::I64(_) => size_of::<i64>(),
        }
    }
}
