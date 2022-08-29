using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class BinaryLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Binary;
        public BinaryLayer(int size, int previousSize, Distribution weightInit = Distribution.Default, Distribution biasInit = Distribution.Default, bool skipInit = false)
            : base
            (
                  size,
                  previousSize,
                  weightInit == Distribution.Default ? Distribution.Normal : weightInit,
                  biasInit == Distribution.Default ? Distribution.Random : biasInit,
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

        
        public override void Derivate(Span<float> output)
        {
            throw new NotImplementedException(); 
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
