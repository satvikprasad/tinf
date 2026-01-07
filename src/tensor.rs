pub const MAX_RANK: usize = 4;

#[derive(Debug, Clone, Copy)]
pub enum TensorType {
    F32,
    F64,
    I64,
}

impl TensorType {
    pub fn elem_size(&self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
            Self::I64 => size_of::<i64>(),
        }
    }
}

pub type TensorShape = [usize; MAX_RANK];

#[derive(Debug)]
pub struct TensorView<'a, T> {
    pub data: &'a mut [T],
    pub shape: TensorShape, // Capped on max_rank = 4
    pub strides: [usize; MAX_RANK],
}

impl<'a, T> TensorView<'a, T> {
    pub fn compute_strides(shape: TensorShape) -> [usize; MAX_RANK] {
        let mut current = 1;
        let mut strides = [1usize; MAX_RANK];

        for i in (0..MAX_RANK).rev() {
            strides[i] = current;
            current *= shape[i];
        }

        strides
    }

    pub fn from_buffer(data: &'a mut [T], shape: TensorShape) -> Self {
        let strides = TensorView::<T>::compute_strides(shape);

        Self {
            data,
            shape,
            strides,
        }
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

    fn from_tensor_unchecked<'a>(tensor: &'a Tensor) -> &'a [Self]
    where
        Self: Sized;

    fn from_tensor_mut<'a>(tensor: &'a mut Tensor) -> Result<&'a mut [Self], String>
    where
        Self: Sized;

    fn from_tensor_mut_unchecked<'a>(tensor: &'a mut Tensor) -> &'a mut [Self]
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
                return Tensor::$variant(TensorView::<$t>::from_buffer(data.as_mut_slice(), shape));
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

            fn from_tensor_unchecked<'a>(tensor: &'a Tensor) -> &'a [Self]
            where
                Self: Sized,
            {
                if let Tensor::$variant(view) = tensor {
                    view.data
                } else {
                    unreachable!();
                }
            }

            fn from_tensor_mut<'a>(tensor: &'a mut Tensor) -> Result<&'a mut [Self], String>
            where
                Self: Sized,
            {
                match tensor {
                    Tensor::$variant(view) => Ok(view.data),
                    _ => Err(format!("Invalid tensor type retrieval")),
                }
            }

            fn from_tensor_mut_unchecked<'a>(tensor: &'a mut Tensor) -> &'a mut [Self]
            where
                Self: Sized,
            {
                if let Tensor::$variant(view) = tensor {
                    view.data
                } else {
                    unreachable!();
                }
            }

            fn as_tensor_sliced<'a>(data: &'a mut [Self], shape: TensorShape) -> Tensor<'a>
            where
                Self: Sized,
            {
                return Tensor::$variant(TensorView::<$t>::from_buffer(data, shape));
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

    pub unsafe fn from_arena(
        arena: *mut u8,
        byte_offset: usize,
        shape: TensorShape,
        r#type: TensorType,
    ) -> Self {
        macro_rules! impl_from_arena {
            ({ $($t:ty => $variant:ident),* $(,)? }) => {
                match r#type {
                    $(
                        TensorType::$variant => {
                            let data = unsafe {
                                std::slice::from_raw_parts_mut(
                                    arena.add(byte_offset) as *mut $t,
                                    shape.iter().product(),
                                )
                            };

                            <$t>::as_tensor_sliced(data, shape)
                        }
                    )*
                }
            };
        }

        impl_from_arena!({
            f32 => F32,
            f64 => F64,
            i64 => I64
        })
    }

    pub fn shape(&self) -> TensorShape {
        match self {
            Self::F32(view) => view.shape,
            Self::I64(view) => view.shape,
            Self::F64(view) => view.shape,
        }
    }

    pub fn strides(&self) -> TensorShape {
        match self {
            Self::F32(view) => view.strides,
            Self::I64(view) => view.strides,
            Self::F64(view) => view.strides,
        }
    }

    pub fn data<T: ToTensor>(&'a self) -> Result<&'a [T], String> {
        T::from_tensor(self)
    }

    pub fn data_unchecked<T: ToTensor>(&'a self) -> &'a [T] {
        T::from_tensor_unchecked(self)
    }

    pub fn data_mut<T: ToTensor>(&'a mut self) -> Result<&'a mut [T], String> {
        T::from_tensor_mut(self)
    }

    pub fn data_mut_unchecked<T: ToTensor>(&'a mut self) -> &'a mut [T] {
        T::from_tensor_mut_unchecked(self)
    }

    pub fn get_type(&self) -> TensorType {
        match self {
            Self::F32(_) => TensorType::F32,
            Self::F64(_) => TensorType::F64,
            Self::I64(_) => TensorType::I64,
        }
    }

    pub fn elem_size(&self) -> usize {
        self.get_type().elem_size()
    }
}

impl From<&Tensor<'_>> for Vec<u8> {
    fn from(value: &Tensor) -> Self {
        match value {
            Tensor::F32(tensor_data) => tensor_data
                .data
                .iter()
                .flat_map(|f| f.to_ne_bytes())
                .collect(),
            Tensor::F64(tensor_data) => tensor_data
                .data
                .iter()
                .flat_map(|f| f.to_ne_bytes())
                .collect(),
            Tensor::I64(tensor_data) => tensor_data
                .data
                .iter()
                .flat_map(|i| i.to_ne_bytes())
                .collect(),
        }
    }
}
