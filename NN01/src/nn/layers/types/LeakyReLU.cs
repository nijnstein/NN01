﻿using System;
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
                // compute sum of weights multiplied with input neurons then add bias
                float value = Intrinsics.Sum(Intrinsics.Multiply(Weights[j], previous.Neurons, values)) + Biases[j];

                // leaky relu 
                //    Neurons[j] = value < 0 ? MathF.Exp(value) - 1 : value;
                Neurons[j] = value < 0 ? 0.01f : value;
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
