using ILGPU;
using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public abstract class Layer
    {
        public float[] Neurons;

        public float[] Delta;   // error * derivate 
        public float[] Gamma;   // derivate of activation


        public float[,] Weights;
        public float[] Biases;

        public int Size;
        public int PreviousSize;

        internal float[,] WeightDeltas;
        internal float[] BiasDeltas;

        public bool IsInput => PreviousSize == 0;
        public bool Softmax { get; set; }

        public abstract LayerActivationFunction ActivationType { get; }
        //
        //
        //      Full - single  (later convolutions)  
        //
        //
        public LayerConnectedness Connectedness { get; set; } = LayerConnectedness.Full;
        public LayerInitializationType WeightInitializer { get; set; } = LayerInitializationType.Random;
        public LayerInitializationType BiasInitializer { get; set; } = LayerInitializationType.dot01;

        public Layer(int size, int previousSize, LayerInitializationType weightInit = LayerInitializationType.Random, LayerInitializationType biasInit = LayerInitializationType.Random, bool softmax = false, bool skipInit = true, IRandom random = null)
        {
            Size = size;
            PreviousSize = previousSize;
            Neurons = new float[size];
            Gamma = new float[size];    // if last then its used to store the activation before softmax if enabled 
            Delta = new float[size]; 

            WeightInitializer = weightInit;
            BiasInitializer = biasInit;
            Softmax = softmax;

            if (!IsInput)
            {
                Biases = new float[size];
                if (!skipInit)
                {
                    InitializeDistribution(BiasInitializer, Biases, random);
                }

                Weights = new float[size, previousSize];
                if (!skipInit)
                {
                    InitializeDistribution(WeightInitializer, Weights.AsSpan2D<float>().Span, random);
                }
            }
            else
            {
                Weights = null!;
                WeightDeltas = null!;
                Biases = null!;
                BiasDeltas = null!;
            }
        }

        public void Activate(Layer previous)
        {
            Activate(previous, previous.Neurons, Neurons);
        }
        public void Derivate(Span<float> output)
        {
            Derivate(Neurons, output); 
        }

        public abstract void Activate(Layer previous, Span<float> inputData, Span<float> outputData);
        public abstract void ReversedActivation(Layer next);

        /// <summary>
        /// gamma == backward activation of neuron -> derivate(state) 
        /// </summary>
        /// <param name="delta"></param>
        /// <param name="gamma"></param>
        /// <param name="target"></param>
        public abstract void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target);
        public abstract void Derivate(Span<float> input, Span<float> output);


        /// <summary>
        /// generate different distributions from a uniform random number generator 
        /// uniform -1..0..1
        /// </summary>
        /// <param name="initializer"></param>
        /// <param name="data"></param>
        /// <param name="random"></param>
        private void InitializeDistribution(LayerInitializationType initializer, Span<float> data, IRandom random)
        {
            switch (initializer)
            {
                case LayerInitializationType.dot01: for (int i = 0; i < data.Length; i++) data[i] = 0.01f; break;
                case LayerInitializationType.Zeros: for (int i = 0; i < data.Length; i++) data[i] = 0; break;
                case LayerInitializationType.Ones: for (int i = 0; i < data.Length; i++) data[i] = 1; break;

                default:
                case LayerInitializationType.Random:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {            
                            data[i] = MathF.Abs(random.NextSingle());
                        }
                    }
                    break;

                case LayerInitializationType.Normal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.NormalKernel(
                                random.NextSingle(), random.NextSingle(), 
                                0, 1, 0));
                        }
                    }
                    break;

                case LayerInitializationType.Glorot:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.GlorotKernel(
                                random.NextSingle(), random.NextSingle(),
                                0f, 1f, Size + PreviousSize));
                        }
                    }
                    break;

                case LayerInitializationType.HeNormal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = MathF.Abs(RandomDistributionInfo.HeNormalKernel(
                                random.NextSingle(), random.NextSingle(),
                                0, 1, Size));
                        }
                    }
                    break;

                case LayerInitializationType.Xavier:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = RandomDistributionInfo.XavierKernel(random.NextSingle(), PreviousSize);
                        }
                    }
                    break;

                case LayerInitializationType.XavierNormalized:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = RandomDistributionInfo.XavierNormalizedKernel(random.NextSingle(), PreviousSize, Size);
                        }
                    }
                    break;

                case LayerInitializationType.Uniform:
                    {
                        MathEx.Uniform(data, 1f);
                    }
                    break;
            }
        }

        public void InitializeDeltaBuffers()
        {
            if (!IsInput)
            {
                BiasDeltas = new float[Size];
                WeightDeltas = (IsInput ? null : new float[Size, PreviousSize])!;
            }
        }

        private void DisposeDeltaBuffers()
        {
            if (WeightDeltas != null) WeightDeltas = null!;
            if (BiasDeltas != null) BiasDeltas = null!;
        }

        public void Update(Layer previous, float[] gamma, float weightLearningRate, float biasLearningRate, float momentum, float weightCost)
        {
            if (BiasDeltas == null || WeightDeltas == null)
            {
                // these may not be initialized if the network is used for inference only 
                InitializeDeltaBuffers();
            }

            Vector256<float> wRate;
            Vector256<float> wCost;
            Span<Vector256<float>> prev;
            if (Avx.IsSupported)
            {
                wRate = Vector256.Create(weightLearningRate);
                wCost = Vector256.Create(weightCost);
                prev = MemoryMarshal.Cast<float, Vector256<float>>(previous.Neurons);
            }
            else
            {
                wRate = default;
                wCost = default;
                prev = default;
            }

            // update bias (avx)
            if (Avx.IsSupported)
            {
                Span<Vector256<float>> b = MemoryMarshal.Cast<float, Vector256<float>>(Biases!);
                Span<Vector256<float>> bDelta = MemoryMarshal.Cast<float, Vector256<float>>(BiasDeltas!);
                Span<Vector256<float>> gi = MemoryMarshal.Cast<float, Vector256<float>>(gamma);
                int i = 0, ii = 0;
                while (i < (Size & ~7))
                {
                    Vector256<float> deltaBias = Avx.Multiply(gi[ii], Vector256.Create(biasLearningRate));
                    b[ii] = Avx.Subtract(b[ii], Avx.Add(deltaBias, Avx.Multiply(bDelta[ii], Vector256.Create(momentum))));
                    bDelta[ii] = deltaBias;
                    ii += 1;
                    i += 8;
                }
                while (i < Size)
                {
                    float delta = gamma[i] * biasLearningRate;
                    Biases[i] -= delta + (BiasDeltas![i] * momentum);
                    BiasDeltas![i] = delta;
                    i++;
                }
            }

            // calculate new weights and biases for the last layer in the network 
            for (int i = 0; i < Size; i++)
            {
                if (!Avx.IsSupported)
                {
                    float delta = gamma[i] * biasLearningRate;
                    Biases[i] -= delta + (BiasDeltas![i] * momentum);
                    BiasDeltas![i] = delta;
                }

                // apply some learning... move in direction of result using gamma 
                int j = 0;
                if (Avx.IsSupported)
                {
                    Vector256<float> g = Vector256.Create(gamma[i]);
                    Vector256<float> gw = Vector256.Create(gamma[i] * weightLearningRate);
                    Span<Vector256<float>> w = MemoryMarshal.Cast<float, Vector256<float>>(Weights.AsSpan2D<float>().Row(i));

                    int jj = 0;
                    unchecked
                    {
                        Span2D<float> wd = WeightDeltas.AsSpan2D<float>();

                        while (j < (previous.Size & ~7))
                        {
                            Span<Vector256<float>> wDelta = MemoryMarshal.Cast<float, Vector256<float>>(wd.Row(i));

                            Vector256<float> d = Avx.Multiply(prev[jj], gw);
                            w[jj] =
                                Avx.Subtract(w[jj],
                                // delta 
                                Avx.Add(
                                    Avx.Add(d, Avx.Multiply(wDelta[jj], Vector256.Create(momentum))),
                                    // strengthen learned weights
//                                    Avx.Multiply(wRate, Avx.Multiply(g, Avx.Multiply(wCost, w[jj])))
                                    Avx.Multiply(wRate, Avx.Subtract(g, Avx.Multiply(wCost, w[jj])))
                                ));

                            wDelta[jj] = d;
                            j += 8;
                            jj++;
                        }
                    }
                }
                unchecked
                {
                    // calc the rest with scalar math
                    while (j < previous.Size)
                    {
                        float delta = previous.Neurons[j] * gamma[i] * weightLearningRate;

                        Weights[i, j] -=
                            // delta 
                            delta
                            // momentum 
                            + (WeightDeltas![i, j] * momentum)
                            // strengthen learned weights
//                            + (weightLearningRate * (gamma[i] * weightCost * Weights[i][j]));  
                              + (weightLearningRate * (gamma[i] - weightCost * Weights[i, j]));  

                        WeightDeltas[i, j] = delta;

                        j++;
                    }
                }
            }
        }

        /// <summary>
        /// the easy to read version
        /// </summary>
        public void UpdateOnCPU(Layer previous, float[] gamma, float weightLearningRate, float biasLearningRate, float momentum, float weightCost)
        {
            // calculate new weights and biases for the last layer in the network 
            for (int i = 0; i < Size; i++)
            {
                float delta = gamma[i] * biasLearningRate;
                Biases[i] -= delta + (BiasDeltas![i] * momentum);
                BiasDeltas![i] = delta;

                // apply some learning... move in direction of result using gamma 
                int j = 0;
                unchecked
                {
                    // calc the rest with scalar math
                    while (j < previous.Size)
                    {
                        delta = previous.Neurons[j] * gamma[i] * weightLearningRate;

                        Weights[i, j] -=
                            // delta 
                            delta
                            // momentum 
                            + (WeightDeltas![i, j] * momentum)
                            // strengthen learned weights
                            + (weightLearningRate * (gamma[i] - weightCost * Weights[i, j]));

                        WeightDeltas[i, j] = delta;

                        j++;
                    }
                }
            }
        }
    }
}
