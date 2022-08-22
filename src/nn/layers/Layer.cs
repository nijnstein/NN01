using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
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

        public LayerInitializer WeightInitializer { get; set; } = LayerInitializer.Random;
        public LayerInitializer BiasInitializer { get; set; } = LayerInitializer.Random;

        public Layer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Random, LayerInitializer biasInit = LayerInitializer.Random, bool skipInit = true)
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

        private void InitializeDistribution(LayerInitializer initializer, float[] data)
        {
            switch (initializer)
            {
                case LayerInitializer.Zeros: for (int i = 0; i < data.Length; i++) data[i] = 0; break;
                case LayerInitializer.Ones: for (int i = 0; i < data.Length; i++) data[i] = 1; break;
                
                default:
                case LayerInitializer.Random:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.NextSingle() - 0.5f;
                        }
                    }
                    break;
                
                case LayerInitializer.Normal:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.Normal(0, 1);
                        }
                    }
                    break; 

                case LayerInitializer.Gaussian:
                    {
                        for (int i = 0; i < data.Length; i++)
                        {
                            data[i] = Random.Shared.Gaussian(0, 1);
                        }
                    }
                    break; 
                
                case LayerInitializer.HeNormal:
                    {
                        if (PreviousSize == 0)
                        {
                            // reduce to random 
                            goto case LayerInitializer.Normal;
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

                case LayerInitializer.Uniform:
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

            // calculate new weights and biases for the last layer in the network 
            for (int i = 0; i < Size; i++)
            {
                float delta = gamma[i] * biasLearningRate;

                Biases[i] -= delta + (momentum * BiasDeltas![i]);
                BiasDeltas[i] = delta; 

                // apply some learning... move in direction of result using gamma 
                for (int j = 0; j < previous.Size; j++)
                {
                    delta = gamma[i] * previous.Neurons[j] * weightLearningRate; 

                    Weights[i][j] -= 
                        delta 
                        // hold momentum 
                        + (momentum * WeightDeltas![i][j]) 
                        // strengthen learned weights
                        + (weightLearningRate * (gamma[i] - weightCost * Weights[i][j]));

                    WeightDeltas[i][j] = delta; 
                }
            }
        }
    }
    

 
}
