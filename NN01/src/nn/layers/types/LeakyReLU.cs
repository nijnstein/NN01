using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class LeakyReLuLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.LeakyReLU;
        public LeakyReLuLayer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Default, LayerInitializer biasInit = LayerInitializer.Default, bool skipInit = false)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializer.Default ? LayerInitializer.HeNormal : weightInit,
                  biasInit == LayerInitializer.Default ? LayerInitializer.Random : biasInit,
                  skipInit
            )
        {
        }
        public override void Activate(Layer previous)
        {
            Span<float> values = stackalloc float[previous.Size];

            for (int j = 0; j < Size; j++)
            {
                // multiply weight with input neuron 
                for (int k = 0; k < previous.Size; k++)
                {
                    values[k] = Weights[j][k] * previous.Neurons[k];
                }

                // column sum
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += values[k];
                }

                // apply bias
                value += Biases[j];
                      
                // leaky relu 
                Neurons[j] = value < 0 ? MathF.Exp(value) - 1 : value;
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (target[i] < 0 ? 0.01f : 1f);
            }
        }
    }

}
