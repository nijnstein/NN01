using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class InputLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.None;
        public InputLayer(int size) : base(size, 0, LayerInitializationType.Zeros, LayerInitializationType.Zeros) { }

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            unchecked
            {
                for (int i = 0; i < inputData.Length; i++)
                {
                   outputData[i] = inputData[i];
                }
            }
        }
        public override void ReversedActivation(Layer next)
        {
            throw new NotImplementedException();
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            throw new NotImplementedException(); 
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            throw new NotImplementedException();
        }
    }
}
