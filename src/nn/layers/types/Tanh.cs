using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class TanhLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Tanh;
        public TanhLayer(int size, int previousSize) : base(size, previousSize, LayerInitializer.HeNormal, LayerInitializer.Random) { }
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
                Neurons[j] = (float)Math.Tanh(value);
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // gamma == difference times  activationDerivative(neuron value)
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * (1 - target[i] * target[i]);
            }
        }
    }
}
