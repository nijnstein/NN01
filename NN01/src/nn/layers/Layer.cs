using ILGPU;
using ILGPU.Runtime;
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
        public float[][] Weights;
        public float[] Biases;

        public int Size;
        public int PreviousSize;

        internal float[][] WeightDeltas;
        internal float[] BiasDeltas;

        public bool IsInput => PreviousSize == 0;

        public abstract LayerActivationFunction ActivationType { get; }
        //
        //
        //      Full - single  (later convolutions)  
        //
        //
        public LayerConnectedness Connectedness { get; set; } = LayerConnectedness.Full;
        public Distribution WeightInitializer { get; set; } = Distribution.Random;
        public Distribution BiasInitializer { get; set; } = Distribution.Random;

        public Layer(int size, int previousSize, Distribution weightInit = Distribution.Random, Distribution biasInit = Distribution.Random, bool skipInit = true)
        {
            Size = size;
            PreviousSize = previousSize;
            Neurons = new float[size];

            WeightInitializer = weightInit;
            BiasInitializer = biasInit;

            if (!IsInput)
            {
                Biases = new float[size];
                if (!skipInit)
                {
                    InitializeDistribution(BiasInitializer, Biases);
                }

                Weights = (IsInput ? null : new float[size][])!;
                for (int i = 0; i < size; i++)
                {
                    Weights![i] = new float[previousSize];
                    if (!skipInit)
                    {
                        InitializeDistribution(WeightInitializer, Weights[i]);
                    }
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

        public abstract void Activate(Layer previous);
        public abstract void CalculateGamma(float[] delta, float[] gamma, float[] target);
        public abstract void Derivate(Span<float> output);

        private void InitializeDistribution(Distribution initializer, float[] data)
        {
            switch (initializer)
            {
                case Distribution.Zeros: for (int i = 0; i < data.Length; i++) data[i] = 0; break;
                case Distribution.Ones: for (int i = 0; i < data.Length; i++) data[i] = 1; break;

                default:
                case Distribution.Random:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.NextSingle() - 0.5f;
                        }
                    }
                    break;

                case Distribution.Normal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.Normal(0, 1);
                        }
                    }
                    break;

                case Distribution.Gaussian:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.Gaussian(0, 1);
                        }
                    }
                    break;

                case Distribution.HeNormal:
                    {
                        if (PreviousSize == 0)
                        {
                            // reduce to random 
                            goto case Distribution.Normal;
                        }
                        else
                        {
                            // use the size of input to weights as base for sd centered around mean 0

                            //float fan_inout = (float)PreviousSize * (float)Math.Sqrt(1f / ((PreviousSize + Size) / 2));
                            float fan = PreviousSize * (float)Math.Sqrt(1f / PreviousSize);

                            for (int i = 0; i < data.Length; i++)
                            {
                                data[i] = Random.Shared.Normal(0, fan);
                            }

                        }
                    }
                    break;

                case Distribution.Uniform:
                    {
                        MathEx.Uniform(data, 1f);
                    }
                    break;
            }
        }

        private void InitializeDeltaBuffers()
        {
            if (!IsInput)
            {
                BiasDeltas = new float[Size];
                BiasDeltas.Zero();
                WeightDeltas = (IsInput ? null : new float[Size][])!;
                for (int i = 0; i < Size; i++)
                {
                    WeightDeltas![i] = new float[PreviousSize];
                    WeightDeltas[i].Zero();
                }
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
                    Span<Vector256<float>> w = MemoryMarshal.Cast<float, Vector256<float>>(Weights[i]);

                    int jj = 0;
                    unchecked
                    {
                        while (j < (previous.Size & ~7))
                        {
                            Span<Vector256<float>> wDelta = MemoryMarshal.Cast<float, Vector256<float>>(WeightDeltas![i]);

                            Vector256<float> d = Avx.Multiply(prev[jj], gw);
                            w[jj] =
                                Avx.Subtract(w[jj],
                                // delta 
                                Avx.Add(
                                    Avx.Add(d, Avx.Multiply(wDelta[jj], Vector256.Create(momentum))),
                                    // strengthen learned weights
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

                        Weights[i][j] -=
                            // delta 
                            delta
                            // momentum 
                            + (WeightDeltas![i][j] * momentum)
                            // strengthen learned weights
                            + (weightLearningRate * (gamma[i] - weightCost * Weights[i][j]));

                        WeightDeltas[i][j] = delta;

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

                        Weights[i][j] -=
                            // delta 
                            delta
                            // momentum 
                            + (WeightDeltas![i][j] * momentum)
                            // strengthen learned weights
                            + (weightLearningRate * (gamma[i] - weightCost * Weights[i][j]));

                        WeightDeltas[i][j] = delta;

                        j++;
                    }
                }
            }
        }
    }
}
