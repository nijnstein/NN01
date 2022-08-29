using ILGPU;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using System;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Dynamic;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using System.Text;
using ILGPU.Algorithms.Random;

namespace NN01
{

    public struct NetworkBuffers<T> where T : unmanaged
    {
        public int LayerIndex { get; init; }
        public bool IsLast { get; init; }

        public Memory<T> Neurons { get; init; }
        public Memory<T> Weights { get; init; }
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
                Weights = default;
                Bias = default;
                WeightDeltas = default;
                BiasDeltas = default;
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

     



    public class NeuralNetwork2 : IComparable, IDisposable
    {
        protected Buffers<float>? buffers = null; 


        public LayerInfo[] Layers { get; protected set; }
        public int LayerCount => Layers == null ? 0 : Layers.Length;
        public float Cost { get; protected set; }
        public float Fitness { get; set; }

        public LayerInfo Input => Layers != null && Layers.Length > 0 ? Layers[0] : default;
        public LayerInfo Output => Layers != null && Layers.Length > 0 ? Layers[Layers.Length - 1] : default;


        public NeuralNetwork2(params LayerInfo[] info)
        {
            Debug.Assert(info != null, "must supply layer info");
            Debug.Assert(info.Length > 2, "there must be at least information for 3 layers: input-[hidden]-output");
            Debug.Assert(info[0].IsInput, "first layer must be an input layer");
            Debug.Assert(info[info.Length - 1].IsOutput, "last layer must be configured as output layer"); 

            Layers = info;

            int[] bufferSizes = Layers.SelectMany(x => x.EnumerateBufferSizes( x.LayerIndex < x.LayerCount - 1 ? Layers[x.LayerIndex + 1] : default)).ToArray();
            buffers = new Buffers<float>(bufferSizes, true); 
        }

        public void Dispose()
        {
            if (buffers != null)
            {
                buffers.Dispose();
            }
        }

        public void Initialize(Accelerator? acc = null)
        {
            Debug.Assert(buffers != null, "internal buffer not initialized");

            buffers.Lease(lease =>
            {
                if (acc == null)
                {
                    Span<NetworkBuffers<float>> layerData = GetLayerData(lease);
                    InitializeCPU(layerData);
                }
                else
                {
                    Span<GPUNetworkBuffers<float>> gpuLayerData = GetGPULayerData(lease, acc);
                    InitializeGPU(gpuLayerData, acc);
                    SynchronizeLayerData(gpuLayerData, acc);
                }
            });
        }
                                                                            
        private void SynchronizeLayerData(Span<GPUNetworkBuffers<float>> layerData, Accelerator acc)
        {
            Debug.Assert(acc != null, "GPU Accelerator NULL");

            acc.Synchronize(); 
            
            for(int i = 0; i < layerData.Length; i++)
            {
                LayerInfo layer = Layers[i];
                GPUNetworkBuffers<float> data = layerData[i];

                data.Neurons.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Neurons.Span);
                data.Weights.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Weights.Span);
                data.WeightDeltas.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.WeightDeltas.Span);
                data.Bias.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Bias.Span);
                data.BiasDeltas.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.BiasDeltas.Span);
                data.Gamma.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Gamma.Span);
            }

            acc.Synchronize();
#if DEBUG 
            Console.WriteLine("first weight = " + layerData[0].ManagedBuffers.Weights.Span[0]);
#endif 
        }

        public int CalculateLayerDataSize()
        {
            int totalBufferSize = 0;
            for (int i = 0; i < Layers.Length; i++)
            {
                totalBufferSize += Layers[i].CalculateLayerBufferSize(i < Layers.Length - 1 ? Layers[i + 1] : default);
            }
            return totalBufferSize;
        }
        private Span<NetworkBuffers<float>> GetLayerData(IBufferLease<float> lease) => Layers.Select(x => lease.GetLayerData(lease, x)).ToArray().AsSpan();
        private Span<GPUNetworkBuffers<float>> GetGPULayerData(IBufferLease<float> lease, Accelerator acc) => Layers.Select(x => lease.GetGPULayerData(lease, x, acc)).ToArray().AsSpan();

        /// <summary>
        /// 32bit floating point processing 
        /// </summary>
        private void InitializeCPU(Span<NetworkBuffers<float>> layerData)
        {
            int previousSize = 0;
            for (int i = 0; i < Layers.Length; i++)
            {
                LayerInfo layer = Layers[i];

                DistributionKernel.GetCPU(layer.BiasInitializer)(layerData[i].Bias, 0f, 1f);
                DistributionKernel.GetCPU(layer.WeightInitializer)(layerData[i].Weights, 0f, 1f);

                DistributionKernel.GetCPU(Distribution.Zeros)(layerData[i].WeightDeltas, 0f, 1f);
                DistributionKernel.GetCPU(Distribution.Zeros)(layerData[i].BiasDeltas, 0f, 1f);
                
                //
                //   need parameters specialized to some generation of distribution (heNormal?) 
                //    
                //   struct params () { low  high  sd mean  size previous} ?  
                //

                previousSize = layer.Size;
            }
        }


        /// <summary>
        /// 32 bit FP processing on GPU 
        /// </summary>
        /// <param name="layerData"></param>
        /// <param name="acc"></param>
        private void InitializeGPU(Span<GPUNetworkBuffers<float>> layerData, Accelerator acc, long seed = 0)
        {
            Debug.Assert(acc != null, "GPU Accelerator NULL"); 

            int previousSize = 0;

            if (seed == 0) seed = DateTime.Now.Ticks;

            using (var rng1 = RNG.Create<XorShift128Plus>(acc, new Random((int)seed)))
            {
                for (int i = 0; i < Layers.Length - 1; i++)
                {
                    LayerInfo layer = Layers[i];

                    var zeroKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(Distribution.Zeros));
                    var biasKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(layer.BiasInitializer));
                    var weightKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(layer.WeightInitializer));

                    if (layer.WeightInitializer != Distribution.Ones && layer.WeightInitializer != Distribution.Zeros)
                    {
                        rng1.FillUniform(layerData[i].Weights.View);
                    }       


                    if (layer.BiasInitializer != Distribution.Ones && layer.BiasInitializer != Distribution.Zeros)
                    {
                        rng1.FillUniform(layerData[i].Bias.View);
                    }

                    weightKernel(acc.DefaultStream, (int)layerData[i].Weights.Length, layerData[i].Weights.View, 0f, 1f);
                    biasKernel(acc.DefaultStream, (int)layerData[i].Bias.Length, layerData[i].Bias.View, 0f, 1f);
                    zeroKernel(acc.DefaultStream, (int)layerData[i].WeightDeltas.Length, layerData[i].WeightDeltas.View, 0, 1f);
                    zeroKernel(acc.DefaultStream, (int)layerData[i].BiasDeltas.Length, layerData[i].BiasDeltas.View, 0, 1f);

                    previousSize = layer.Size;
                }
            }
        }

        static float[] FeedForward(Span<NetworkBuffers<float>> layerData, Span<float> sample)
        {
            Debug.Assert(sample.Length == layerData[0].Neurons.Length, "sample size does not match input size");

            // copy input neurons into layerdata 
            Span<float> input = layerData[0].Neurons.Span;

            for (int i = 0; i < sample.Length; i++)
            {
                input[i] = sample[i];
            }

            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        static float[] FeedForward(Span<GPUNetworkBuffers<float>> layerData, Span<float> sample, Accelerator acc)
        {
            Debug.Assert(sample.Length == layerData[0].Neurons.Length, "sample size does not match input size");

            // copy input neurons to GPU into layerdata 
            layerData[0].Neurons.View.CopyFromCPU(acc.DefaultStream, sample); 
                           
            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        public void Train( )
        {
            Debug.Assert(buffers != null, "internal buffer not initialized"); 

            buffers.Lease( lease =>
            {
                for(int i = 0; i < Layers.Length; i++)
                {
                

                }
            });
        }





        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");
            for (int i = 0; i < Layers.Length; i++)
            {
                sb.Append($"{Layers[i].Size}");
                if (i < Layers.Length - 1)
                {
                    sb.Append($"-{Layers[i].ActivationType}-");
                }
            }
            sb.Append("]");
            return sb.ToString();
        }
        

     

        public int CompareTo(object? obj)
        {
            return CompareTo((obj as NeuralNetwork2)!);
        }

        public int CompareTo(NeuralNetwork2 other)
        {
            if (other == null) return 1;

            if (Fitness > other.Fitness)
                return 1;
            else if (Fitness < other.Fitness)
                return -1;
            else
                return 0;
        }
    }
}

