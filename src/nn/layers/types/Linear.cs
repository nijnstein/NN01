using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class LinearLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Linear;
        public LinearLayer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Default, LayerInitializer biasInit = LayerInitializer.Default, bool skipInit = false)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializer.Default ? LayerInitializer.Ones : weightInit,
                  biasInit == LayerInitializer.Default ? LayerInitializer.Zeros : biasInit,
                  skipInit
            )
        {
        }
        public override void Activate(Layer previous)
        {
            for (int j = 0; j < Size; j++)
            {
                for (int k = 0; k < previous.Size; k++)
                {
                    Neurons[k] = Weights[j][k] * previous.Neurons[k];
                }
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * target[i];
            }
        }
    }
}
