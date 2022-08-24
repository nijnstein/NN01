using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class BinaryLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Binary;
        public BinaryLayer(int size, int previousSize, LayerInitializer weightInit = LayerInitializer.Default, LayerInitializer biasInit = LayerInitializer.Default, bool skipInit = false)
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
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j][k] * previous.Neurons[k];
                }

                value += Biases[j];

                Neurons[j] = value < 0 ? 0 : 1;
            }
        }
        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            // a binary activation has no gradient, thus in the backward pass an STE (straight through estimator) is used 
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * target[i].HardTanH(-0.5f, 0.5f);
            }
        }
    }

}
