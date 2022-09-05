using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NSS;

namespace NN01
{
    public struct NetworkBuffers<T> where T : unmanaged
    {
        public int LayerIndex { get; init; }
        public bool IsLast { get; init; }

        public Memory<T> Neurons { get; init; }
        public Memory<T> Weights { get; init; }
        
        /// <summary>
        /// the current layer holds the bias needed to activate the next layer 
        /// </summary>
        public Memory<T> Bias { get; init; }

        public Memory<T> WeightDeltas { get; init; }
        public Memory<T> BiasDeltas { get; init; }
        public Memory<T> Gamma { get; init; }

        private int Index(int id) => (LayerIndex * LayerInfo.BufferCount) + id;

        public NetworkBuffers(IBufferLease<T> lease, int layerIndex, bool islast = false)
        {
            // calc baseindex then add id|index to get location of span in alignedbuffer<T> 
            int idx(int id) => (layerIndex * LayerInfo.BufferCount) + id;

            LayerIndex = layerIndex;
            IsLast = islast;

            Neurons = lease.GetMemory(idx(LayerInfo.NeuronBufferId));
            if (!islast)
            {
                Weights = lease.GetMemory(idx(LayerInfo.WeightBufferId));
                Bias = lease.GetMemory(idx(LayerInfo.BiasBufferId));
                WeightDeltas = lease.GetMemory(idx(LayerInfo.WeightDeltaBufferId));
                BiasDeltas = lease.GetMemory(idx(LayerInfo.BiasDeltaBufferId));
            }
            else 
            {
                Weights = Memory<T>.Empty;
                Bias = Memory<T>.Empty;
                WeightDeltas = Memory<T>.Empty;
                BiasDeltas = Memory<T>.Empty; 
            }
            Gamma = lease.GetMemory(idx(LayerInfo.GammaBufferId));
        }
    }


    public struct GPUNetworkBuffers<T> where T : unmanaged
    {   
        public NetworkBuffers<T> ManagedBuffers;

        public MemoryBuffer1D<float, Stride1D.Dense> Neurons;

        public MemoryBuffer1D<float, Stride1D.Dense> Gamma;
        public MemoryBuffer1D<float, Stride1D.Dense> Weights;
        public MemoryBuffer1D<float, Stride1D.Dense> Bias;

        public MemoryBuffer1D<float, Stride1D.Dense> WeightDeltas;
        public MemoryBuffer1D<float, Stride1D.Dense> BiasDeltas;

        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> ActivationKernel;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int> DerivateKernel; 
        
        public int LayerIndex => ManagedBuffers.LayerIndex;
        public bool IsLast => ManagedBuffers.IsLast;

        public GPUNetworkBuffers(IBufferLease<T> lease, LayerInfo info, Accelerator acc)
        {
            Debug.Assert(acc != null, "GPU Accelerator == null");

            ManagedBuffers = new NetworkBuffers<T>(lease, info.LayerIndex, info.LayerIndex == info.LayerCount - 1);

            Neurons = acc.Allocate1D<float>(ManagedBuffers.Neurons.Length);
            Weights = acc.Allocate1D<float>(ManagedBuffers.Weights.Length);
            Bias = acc.Allocate1D<float>(ManagedBuffers.Bias.Length);
            WeightDeltas = acc.Allocate1D<float>(ManagedBuffers.WeightDeltas.Length);
            BiasDeltas = acc.Allocate1D<float>(ManagedBuffers.BiasDeltas.Length);
            Gamma = acc.Allocate1D<float>(ManagedBuffers.Gamma.Length);

            if (info.ActivationType != LayerActivationFunction.None)
            {
                //ActivationKernel = acc.LoadKernel(NN01.ActivationKernel.GetGPU(info.ActivationType, false));
                ActivationKernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>
                (
                    NN01.ActivationKernel.GetGPU(info.ActivationType)
                );

                DerivateKernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int>(
                    NN01.ActivationKernel.GetDerivateGPU(info.ActivationType)
                );
            }
            else
            {
                ActivationKernel = null!;
                DerivateKernel = null!;
            }
        }

        public void Dispose()
        {
            Neurons.Dispose();
            Weights.Dispose();
            Bias.Dispose();
            WeightDeltas.Dispose();
            BiasDeltas.Dispose();
            Gamma.Dispose();
        }
    }
}
