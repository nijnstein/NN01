using ILGPU;
using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{

    public abstract class ParameterLayer : Layer
    {
        public float[,] Weights;
        public float[] Biases;

        public float[,] WeightDeltas;
        public float[] BiasDeltas;

        public override bool HasParameters => true; 
        public abstract LayerInitializationType WeightInitializer { get; }
        public abstract LayerInitializationType BiasInitializer { get; }

        public ParameterLayer(int size, int previousSize, bool skipInit = true, IRandom random = null)
            : base(size, previousSize)
        {
            Biases = new float[size];
            Weights = new float[size, previousSize];
            BiasDeltas = new float[Size];
            WeightDeltas = (IsInput ? null : new float[Size, PreviousSize])!;
            if (!skipInit)
            {
                InitializeParameters(random);
            }
        }
 
        public override void Update(Layer previous, float[] gamma, float weightLearningRate, float biasLearningRate, float momentum, float weightCost)
        {
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
                                    // Avx.Multiply(wRate, Avx.Multiply(g, Avx.Multiply(wCost, w[jj])))
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


        public void InitializeParameters(IRandom? random = null)
        {
            bool ownRandom = random == null;
            if (ownRandom)
            {
                random = new CPURandom(RandomDistributionInfo.Uniform(0, 1f));
            }

            if (Biases != null)
            {
                FillParameterDistribution(BiasInitializer, Biases, random!);
            }

            if (Weights != null)
            {
                FillParameterDistribution(WeightInitializer, Weights.AsSpan2D<float>().Span, random!);
            }

            if (ownRandom)
            {
                random!.Dispose();
            }
        }
    }
}
