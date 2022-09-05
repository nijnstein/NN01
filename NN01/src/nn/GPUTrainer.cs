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
using System.Runtime.CompilerServices;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Algorithms;
using NSS;
using NSS.Neural;

namespace NN01
{


    public class GPUTrainer : IComparable, IDisposable
    {

        protected Buffers256<float>? buffers = null;
        

        /// <summary>
        /// 
        /// 
        ///     move aligned buffer use to neuralnetwork 1 code and make this one only a host to calculate nn1 fast
        /// 
        /// 
        /// </summary>


        public struct nnlayer
        {  
            // neurons[population, layerSize ]
            // weight [population, n0.size, n1.size]
            public ArrayView3D<float, Stride3D.DenseZY> Weights;
            // bias   [population, n1.size]
            public ArrayView2D<float, Stride2D.DenseX> Bias;
            // gamma  [population, n1.index ]
            public ArrayView2D<float, Stride2D.DenseX> Gamma;
            // delta  [n1.index ]
            public ArrayView2D<float, Stride2D.DenseX> Delta;

            // weight delta [n0.index, n1.index]
            public ArrayView3D<float, Stride3D.DenseZY> WeightDelta;
            // bias delta [n1.index]
            public ArrayView2D<float, Stride2D.DenseX> BiasDelta;

            public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> ActivationKernel;
            public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int> DerivateKernel;
        }

        public struct nn
        {
            public const int LayerCount = 4;
            public int PopulationCount;

            // neuron state and transfer data 

            public ArrayView2D<float, Stride2D.DenseX> n0;
            public nnlayer l01;
            public ArrayView2D<float, Stride2D.DenseX> n1;
            public nnlayer l12;
            public ArrayView2D<float, Stride2D.DenseX> n2;
            public nnlayer l13;
            public ArrayView2D<float, Stride2D.DenseX> n3;

            static public void Initialize(NeuralNetwork network, int populationCount = 128)
            {
                Debug.Assert(network != null && network.LayerCount == LayerCount); 


            }

        }
        

   
        static void ForwardActivation(Index1D d, ArrayView<float> n, nnlayer l, ArrayView<float> p)
        {
            // can we do this in 1 kernel? 

        }


        static void Train(nn network)
        {

         
        }



        public LayerInfo[] Layers { get; protected set; }
        public int LayerCount => Layers == null ? 0 : Layers.Length;
        public float Cost { get; protected set; }
        public float CostDelta { get; protected set; }
        public float Fitness { get; set; }

        public LayerInfo Input => Layers != null && Layers.Length > 0 ? Layers[0] : default;
        public LayerInfo Output => Layers != null && Layers.Length > 0 ? Layers[Layers.Length - 1] : default;


        public GPUTrainer(params LayerInfo[] info)
        {
            Debug.Assert(info != null, "must supply layer info");
            Debug.Assert(info.Length > 2, "there must be at least information for 3 layers: input-[hidden]-output");
            Debug.Assert(info[0].IsInput, "first layer must be an input layer");
            Debug.Assert(info[info.Length - 1].IsOutput, "last layer must be configured as output layer"); 

            Layers = info;

            int[] bufferSizes = Layers.SelectMany(x => x.EnumerateBufferSizes( x.LayerIndex < x.LayerCount - 1 ? Layers[x.LayerIndex + 1] : default)).ToArray();
            buffers = new Buffers256<float>(bufferSizes, true); 
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
                    Span<NetworkBuffers<float>> layerData = default;// GetLayerData(lease);
                    InitializeCPU(layerData);
                }
                else
                {
                    Span<GPUNetworkBuffers<float>> gpuLayerData = default; //GetGPULayerData(lease, acc);
                    InitializeGPU(gpuLayerData, acc);
                    SynchronizeLayerData(gpuLayerData, acc);
                }
            });
        }

        public void Train(SampleSet samples, Accelerator? acc = null)
        {
            Debug.Assert(buffers != null, "internal buffer not initialized");

            float[] sample = new float[Layers[0].Size]; 

            // lease a buffer, provides spans pinned on 256 boundary
            buffers.Lease(lease =>
            {
                if (acc == null)
                {
                 //   for(int i = 0; i < 10; i++) TrainCPU(GetLayerData(lease), samples); 
                }
                else
                {
              //      Span<GPUNetworkBuffers<float>> gpuLayerData = GetGPULayerData(lease, acc);
                 //   using (var sampleData = samples.CopySamplesToGPU(acc))
                  //  using (var expectations = samples.CopyExpectationsToGPU(acc))
                    {
                //        for (int i = 0; i < 10; i++) TrainGPU(gpuLayerData, sampleData, expectations, acc);
                    }
                  //  SynchronizeLayerData(gpuLayerData, acc);
                }
            });
        }

        private void TrainGPU(Span<GPUNetworkBuffers<float>> layerData, ArrayView2D<float, Stride2D.DenseX> sampleData, ArrayView2D<float, Stride2D.DenseX> expectation, Accelerator acc)
        {
            BackPropGPU(layerData, sampleData, expectation, acc);
        }

        private void TrainCPU(Span<NetworkBuffers<float>> layerData, SampleSet sampleData)
        {
            FeedForwardCPU(layerData, sampleData);
        }



        private void SynchronizeLayerData(Span<GPUNetworkBuffers<float>> layerData, Accelerator acc, bool neuronsOnly = true)
        {
            Debug.Assert(acc != null, "GPU Accelerator NULL");
            
            for(int i = 0; i < layerData.Length; i++)
            {
                LayerInfo layer = Layers[i];
                GPUNetworkBuffers<float> data = layerData[i];

                if (neuronsOnly)
                {
                    data.Neurons.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Neurons.Span);
                }
                else
                {
                    data.Weights.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Weights.Span);
                    data.WeightDeltas.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.WeightDeltas.Span);
                    data.Bias.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Bias.Span);
                    data.BiasDeltas.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.BiasDeltas.Span);
                    data.Gamma.View.CopyToCPU(acc.DefaultStream, data.ManagedBuffers.Gamma.Span);
                }
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
//        private Span<NetworkBuffers<float>> GetLayerData(IBufferLease<float> lease) => Layers.Select(x => lease.GetLayerData(lease, x)).ToArray().AsSpan();
        //private Span<GPUNetworkBuffers<float>> GetGPULayerData(IBufferLease<float> lease, Accelerator acc) => Layers.Select(x => lease.GetGPULayerData(lease, x, acc)).ToArray().AsSpan();

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

                DistributionKernel.GetCPU(LayerInitializationType.Zeros)(layerData[i].WeightDeltas, 0f, 1f);
                DistributionKernel.GetCPU(LayerInitializationType.Zeros)(layerData[i].BiasDeltas, 0f, 1f);
                
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

            // initialize layerdata 
            using (var rng1 = RNG.Create<XorShift128Plus>(acc, new Random((int)seed)))
            {                   
                // initialize each layer;s memory 
                for (int i = 0; i < Layers.Length - 1; i++)
                {
                    LayerInfo layer = Layers[i];

                    var zeroKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(LayerInitializationType.Zeros));
                    var biasKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(layer.BiasInitializer));
                    var weightKernel = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float, float>(DistributionKernel.GetGPU(layer.WeightInitializer));

                    if (layer.WeightInitializer != LayerInitializationType.Ones && layer.WeightInitializer != LayerInitializationType.Zeros)
                    {
                        rng1.FillUniform(layerData[i].Weights.View);
                    }       

                    if (layer.BiasInitializer != LayerInitializationType.Ones && layer.BiasInitializer != LayerInitializationType.Zeros)
                    {
                        rng1.FillUniform(layerData[i].Bias.View);
                    }

                    weightKernel(acc.DefaultStream, (int)layerData[i].Weights.Length, layerData[i].Weights.View, 0f, 1f);
                    biasKernel(acc.DefaultStream, (int)layerData[i].Bias.Length, layerData[i].Bias.View, 0f, 1f);
                    zeroKernel(acc.DefaultStream, (int)layerData[i].WeightDeltas.Length, layerData[i].WeightDeltas.View, 0, 1f);
                    zeroKernel(acc.DefaultStream, (int)layerData[i].BiasDeltas.Length, layerData[i].BiasDeltas.View, 0, 1f);

                    previousSize = layer.Size;
                    acc.Synchronize();
                }
            }
        }

      

        private void FeedForwardCPU(Span<NetworkBuffers<float>> layerData, SampleSet sampleData)
        {
          /*  Debug.Assert(sampleData.SampleSize == layerData[0].Neurons.Length, "sample size does not match input size");

            NetworkBuffers<float> outputLayerData = layerData[layerData.Length - 1];

            for (int sampleIndex = 0; sampleIndex < sampleData.Count; sampleIndex++)
            {
                // set input sample 
                Span<float> n0 = layerData[0].Neurons.Span;
                for (int i = 0; i < sampleData.SampleSize; i++)
                {
                    n0[i] = sampleData[sampleIndex].Data[i];
                }

                // get expected labels for sample 
                Span<float> output = outputLayerData.Neurons.Span;
                Span<float> expected = sampleData[sampleIndex].GetExpectationLabels(output.Length);

                // propagate state through layers 
                for (int i = 0; i < layerData.Length - 1; i++)
                {
                    ActivationKernel.GetCPU(Layers[i].ActivationType, false)(layerData[i], layerData[i + 1]);
                }

                // calculate delta and cost 
                float cost = 0;
                float[] delta = new float[outputLayerData.Neurons.Length];
                
                for (int i = 0; i < delta.Length; i++)
                {
                    // precalculate delta 
                    delta[i] = output[i] - expected[i];

                    // calculate cost of network 
                    cost += delta[i] * delta[i];
                }
                CostDelta = MathF.Abs(Cost - cost);
                Cost = cost;
            }*/
        }


        /// <summary>
        /// copy a sample to the input and feed if forward 
        /// </summary>
        private void BackPropGPU(Span<GPUNetworkBuffers<float>> layerData, ArrayView2D<float, Stride2D.DenseX> sampleData, ArrayView2D<float, Stride2D.DenseX> expectation, Accelerator acc)
        {
            Debug.Assert(acc != null);
            Debug.Assert(sampleData.Extent.Y == layerData[0].Neurons.Length, "sample size does not match input size");

            GPUNetworkBuffers<float> outputLayerData = layerData[layerData.Length - 1];
            int layerCount = layerData.Length; 

            // allocate a cost vector to keep cost for each sample 
            ArrayView1D<float, Stride1D.Dense> cost = acc.Allocate1D<float>(sampleData.Extent.X);
            ArrayView1D<float, Stride1D.Dense> delta = acc.Allocate1D<float>(outputLayerData.Neurons.Length);

            // a kernel for copying samples into the first neuron
            var sampleCopier = acc.LoadAutoGroupedKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int>
            (
                (x, samples, neurons, sampleIndex)
                =>
                neurons[x] = samples[sampleIndex, x]
            );

            // delta calculation == delta of output to sample
            var deltaCalculation = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int>
            (
                (x, delta, output, expected, sampleIndex)
                =>
                {
                    delta[x] = output[x] - expected[sampleIndex, x];
                }
            );


            // cost calculation kernel => sum of squares 
            var resetCost = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>>((x, cost) => cost[x] = 0f);
            var costCalculation = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, int>
            (
                (x, delta, cost, sampleIndex)
                =>
                {
                    cost[sampleIndex] += delta[x] * delta[x];
                }
            );
                

            var updateBias = acc.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>
            (
                (i, gamma, bias, biasDelta, biasLearningRate, momentum)
                =>
                {
                    float delta = gamma[i] * biasLearningRate;
                    bias[i] -= delta + (biasDelta![i] * momentum);
                    biasDelta![i] = delta;
                }
            );

            // for i , j 
            var updateWeights = acc.LoadAutoGroupedKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float, float>
            (
                (ij, prev, gamma, weights, weightDelta, weightLearningRate, weightCost, momentum)
                =>
                {
                    float delta = prev[ij.Y] * gamma[ij.X] * weightLearningRate;

                    int index = (int)(ij.X * gamma.Length + ij.Y);

                    weights[index] -=
                        // delta 
                        delta
                        // momentum 
                        + (weightDelta![index] * momentum)
                        // strengthen learned weights
                        + (weightLearningRate * (gamma[ij.X] - weightCost * weights[index]));

                    weightDelta[index] = delta;
                }
            );

            // propagate state through layers 
            resetCost(acc.DefaultStream, (int)cost.Length, cost);

            for (int sampleIndex = 0; sampleIndex < sampleData.Extent.X; sampleIndex++)
            {
                sampleCopier(acc.DefaultStream, (int)sampleData.Extent.Y, sampleData, layerData[0].Neurons, sampleIndex);

                for (int i = 0; i < Layers.Length - 1; i++)
                {
                    var n0 = layerData[i].Neurons.View;
                    var n1 = layerData[i + 1].Neurons.View;
                    var w01 = layerData[i].Weights.View;
                    var b1 = layerData[i].Bias.View;

                    // feed trough
                    layerData[i].ActivationKernel((int)n1.Length, n0, n1, w01, b1);
                }

                // calc delta sample/output/expection
                deltaCalculation(acc.DefaultStream, (int)outputLayerData.Neurons.Length, delta, outputLayerData.Neurons.View, expectation, sampleIndex);

                // determine cost (for each sample)
                costCalculation(acc.DefaultStream, (int)outputLayerData.Neurons.Length, delta, cost, sampleIndex);

                // calculate gamma from transfer of weight from secondlast to outputlayer from the delta
                layerData[layerData.Length - 2].DerivateKernel((int)outputLayerData.Neurons.Length, delta, layerData[layerData.Length - 2].Gamma.View, expectation, sampleIndex);

                // update weights and bias to last
                var layer = layerData[layerData.Length - 2];

                updateBias(acc.DefaultStream, (int)outputLayerData.Neurons.Length, layer.Gamma.View, layer.Bias.View, layer.BiasDeltas.View, 0.01f, 0.01f);
                updateWeights(
                    acc.DefaultStream,
                    new Index2D((int)outputLayerData.Neurons.Length, (int)layer.Neurons.Length),
                    layer.Neurons.View, 
                    layer.Gamma.View,
                    layer.Weights.View,
                    layer.WeightDeltas.View, 0.01f, 0.00001f, 0.01f);
             
             /*   for (int layerIndex = layerCount - 2; layerIndex > 0; layerIndex--)
                {
                    // update gamma from layer weights and current gamma on output
                    layer = layerData[layerIndex];
                    var prev = layerData[layerIndex - 1]; 
                    

                    layer.DerivateKernel(layer.Neurons.Length, layer.Gamma.View, prev.Gamma.View, layer.Neurons, sampleIndex)

                    for (int j = 0; j < layer.Neurons.Length; j++)
                    {


                        

                        // 
                        //  get weighed sum of gamma * previous neurons 
                        // 
                        int k = 0;
                        gamma[layerIndex][j] = 0;
                        while (k < gamma[layerIndex + 1].Length)
                        {
                            gamma[layerIndex][j] += gamma[layerIndex + 1][k] * layers[layerIndex + 1].Weights[k][j];
                            k++;
                        }
                        //
                        // Calculate the new gamma from the activation derivate
                        //
                        layers[layerIndex].CalculateGamma(gamma[layerIndex], gamma[layerIndex], layers[layerIndex].Neurons);
                    }

                    // update layer weights and biases
                    layers[layerIndex].Update(layers[layerIndex - 1], gamma[layerIndex], weightLearningRate, biasLearningRate, momentum, weightCost);
                }
             */
            }
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
            return CompareTo((obj as GPUTrainer)!);
        }

        public int CompareTo(GPUTrainer other)
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

