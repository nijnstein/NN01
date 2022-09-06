using ILGPU.Runtime;
using NSS;
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
        public BinaryLayer(int size, int previousSize, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool softmax = false, bool skipInit = false, IRandom random = null)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializationType.Default ? LayerInitializationType.Normal : weightInit,
                  biasInit == LayerInitializationType.Default ? LayerInitializationType.Random : biasInit,
                  softmax,
                  skipInit, 
                  random
            )
        {
        }
        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            for (int j = 0; j < Size; j++)
            {
                float value = 0f;
                for (int k = 0; k < previous.Size; k++)
                {
                    value += Weights[j, k] * inputData[k];
                }

                value += Biases[j];

                outputData[j] = value < 0 ? 0 : 1;
            }
        }

        public override void ReversedActivation(Layer next)
        {
            throw new NotFiniteNumberException(); 
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            throw new NotImplementedException(); 
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            // a binary activation has no gradient, thus in the backward pass an STE (straight through estimator) is used 
            for (int i = 0; i < Size; i++)
            {
                gamma[i] = delta[i] * target[i].HardTanH(-0.5f, 0.5f);
            }
        }
    }

}
