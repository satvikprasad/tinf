# Tinf
A Rust-based tiny inference engine compatible with ONNX file formats. Designed to be lightweight and efficient on modern CPUs.

## Supported operations
Tinf currently supports the following elementary tensor ops:
- **MatMul**: Matrix multiplication with broadcasting support
- **Conv**: 2D convolution with configurable padding, stride, dilation, and groups
- **Add**: Element-wise addition with broadcasting
- **ReLU**: Rectified linear unit activation
- **MaxPool**: 2D max pooling with configurable kernel size, stride, padding, and dilation
- **Reshape**: Tensor reshaping to specified dimensions

## Benchmarks
|Model|Input Shape|Tinf (ms / inference)|
|MNIST MLP| [1, 784] | 0.73 |

*Benchmarks run on Apple M4, singlethreaded.*

## Roadmap
- [ ] BatchNormalization (with folding into Conv)
- [ ] GlobalAveragePool
- [ ] Sigmoid, Tanh, Softmax activations
- [ ] Concat, Split, Transpose
- [ ] Multi-threaded inference
- [ ] SIMD vectorization
- [ ] INT8 quantization support
