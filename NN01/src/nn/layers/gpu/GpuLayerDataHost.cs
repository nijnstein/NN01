using ILGPU;
using ILGPU.Runtime;

namespace NN01
{
    public class GpuLayerDataHost : IDisposable
    {
        public int Population;
        public int Size;
        public int PreviousSize;
        public bool IsAllocated; 
        public LayerType Type; 

        public MemoryBuffer2D<float, Stride2D.DenseX> Neurons;
        public MemoryBuffer2D<float, Stride2D.DenseX> Delta;   // error * derivate 
        public MemoryBuffer2D<float, Stride2D.DenseX> Gamma;   // derivate of activation

        public MemoryBuffer3D<float, Stride3D.DenseXY> Weights;
        public MemoryBuffer2D<float, Stride2D.DenseX> Biases;
        public MemoryBuffer3D<float, Stride3D.DenseXY> WeightDeltas;
        public MemoryBuffer2D<float, Stride2D.DenseX> BiasDeltas;

        static public GpuLayerDataHost AllocatePopulation(Accelerator acc, Layer layer, int populationCount)
        {
            GpuLayerDataHost gpuLayer = new GpuLayerDataHost(); 
            gpuLayer.Population = populationCount;
            gpuLayer.Size = layer.Size; 
            gpuLayer.PreviousSize = layer.PreviousSize;
            gpuLayer.Type = layer.ActivationType;

            // alloc state
            gpuLayer.Neurons = acc.Allocate2DDenseX<float>(new Index2D(populationCount, gpuLayer.Size));
            gpuLayer.Delta   = acc.Allocate2DDenseX<float>(new Index2D(populationCount, gpuLayer.Size));
            gpuLayer.Gamma   = acc.Allocate2DDenseX<float>(new Index2D(populationCount, gpuLayer.Size));

            // alloc parameters 
            if (layer is ParameterLayer param)
            {
                gpuLayer.Weights = acc.Allocate3DDenseXY<float>(new Index3D(populationCount, gpuLayer.Size, gpuLayer.PreviousSize));
                gpuLayer.WeightDeltas = acc.Allocate3DDenseXY<float>(new Index3D(populationCount, gpuLayer.Size, gpuLayer.PreviousSize));
                gpuLayer.Biases = acc.Allocate2DDenseX<float>(new Index2D(populationCount, gpuLayer.Size));
                gpuLayer.BiasDeltas = acc.Allocate2DDenseX<float>(new Index2D(populationCount, gpuLayer.Size));
            }

            gpuLayer.IsAllocated = true; 
            return gpuLayer;
        }

        public void Dispose()
        {
            if (IsAllocated)
            {
                ReleaseAllocation();
            }
        }

        public void ReleaseAllocation()
        {
            Neurons.Dispose();
            Delta.Dispose();
            Gamma.Dispose();

            if (Type.HasParameters())
            {
                Weights.Dispose();
                WeightDeltas.Dispose();
                Biases.Dispose();
                BiasDeltas.Dispose(); 
            }

            IsAllocated = false; 
        }
    }
}
