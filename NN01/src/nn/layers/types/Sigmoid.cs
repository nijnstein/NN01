using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class SigmoidLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Sigmoid;
        public SigmoidLayer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Default, LayerInitializer biasInit = LayerInitializer.Default, bool skipInit = false)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializer.Default ? LayerInitializer.Normal : weightInit,
                  biasInit == LayerInitializer.Default ? LayerInitializer.Random : biasInit,
                  skipInit
            )
        {
        }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                // weighted sum of (w[j][k] * n[i][k])
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                // apply bias
                value += Biases[j];

                // sigmoid activation  
                float f = MathF.Exp(value);
                Neurons[j] = f / (1.0f + f);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (target[i] * (1f - target[i]));
            }
        }
    }
}
