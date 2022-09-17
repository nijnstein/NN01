using ILGPU;
using ILGPU.Runtime;

namespace NN01
{
    public struct gpu_layer_data
    {
        public int Population;
        public int Size;
        public int PreviousSize;
        public LayerType Type; 

        public ArrayView2D<float, Stride2D.DenseX> Neurons;
        public ArrayView2D<float, Stride2D.DenseX> Delta;   // error * derivate 
        public ArrayView2D<float, Stride2D.DenseX> Gamma;   // derivate of activation
        public ArrayView3D<float, Stride3D.DenseXY> Weights;
        public ArrayView2D<float, Stride2D.DenseX> Biases;

        public ArrayView3D<float, Stride3D.DenseXY> WeightDeltas;
        public ArrayView2D<float, Stride2D.DenseX> BiasDeltas;
    }
}
