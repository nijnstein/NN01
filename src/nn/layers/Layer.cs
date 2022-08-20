using System;
using System.Collections.Generic;
using System.Linq;
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

        public bool IsInput => PreviousSize == 0;

        public abstract LayerActivationFunction ActivationType { get; }

        public LayerInitializer WeightInitializer { get; set; } = LayerInitializer.Random;
        public LayerInitializer BiasInitializer { get; set; } = LayerInitializer.Random;

        public Layer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Random, LayerInitializer biasInit = LayerInitializer.Random)
        {
            Size = size;
            PreviousSize = previousSize;
            Neurons = new float[size];

            WeightInitializer = weightInit;
            BiasInitializer = biasInit;

            if (!IsInput)
            {
                Biases = new float[size];
                InitializeDistribution(BiasInitializer, Biases);

                Weights = (IsInput ? null : new float[size][])!;
                for (int i = 0; i < size; i++)
                {
                    Weights![i] = new float[previousSize];
                    InitializeDistribution(WeightInitializer, Weights[i]);
                }
            }
            else
            {
                Biases = null!;
                Weights = null!;
            }
        }

        public abstract void Activate(Layer previous);
        public abstract void CalculateGamma(float[] delta, float[] gamma, float[] target);

        public void CalculateGamma(float[] gamma)
        {
            CalculateGamma(gamma, gamma, Neurons);
        }

        private void InitializeDistribution(LayerInitializer initializer, float[] data)
        {
            switch (initializer)
            {
                case LayerInitializer.Zeros: for (int i = 0; i < data.Length; i++) data[i] = 0; break;
                case LayerInitializer.Ones: for (int i = 0; i < data.Length; i++) data[i] = 1; break;
                default:
                case LayerInitializer.Random: for (int i = 0; i < data.Length; i++) data[i] = Random.Shared.NextSingle() - 0.5f; break;
                case LayerInitializer.Normal: for (int i = 0; i < data.Length; i++) data[i] = Random.Shared.Normal(0, 1); break;
                case LayerInitializer.Gaussian: for (int i = 0; i < data.Length; i++) data[i] = Random.Shared.Gaussian(0, 1); break;
                case LayerInitializer.HeNormal:

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

                        break;
                    }
            }
        }
    }
    

 
}
