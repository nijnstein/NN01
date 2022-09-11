using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NSS;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class SigmoidLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Sigmoid;
        public SigmoidLayer(int size, int previousSize, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool softmax = false, bool skipInit = false, IRandom random = null)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializationType.Default ? LayerInitializationType.Glorot : weightInit,
                  biasInit == LayerInitializationType.Default ? LayerInitializationType.dot01 : biasInit,
                  softmax,
                  skipInit, 
                  random
            )
        {
        }
        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span<float> values = stackalloc float[previous.Size];
            Span2D<float> w = Weights.AsSpan2D<float>();

            Span<float> values2 = stackalloc float[Size];
            Span<float> values3 = stackalloc float[Size];

            unchecked
            {
                for (int j = 0; j < Size; j++)
                {
                    // weighted sum of (w[j][k] * n[i][k])
                    // apply bias
                    values2[j] = Intrinsics.Sum(Intrinsics.Multiply(w.Row(j), inputData, values)) + Biases[j];
                }
            }

            Intrinsics.Exp(values2, values2);
            Intrinsics.AddScalar(values2, 1, outputData);
            Intrinsics.Divide(values2, outputData, outputData);
            // for (int j = 0; j < Size; j++)
            // {
            //     // sigmoid activation
            //     float f = values2[j];
            //     outputData[j] = f / (1.0f + f);
            // }            
        }

        public override void ReversedActivation(Layer next)
        {
            throw new NotImplementedException();
        }


        public override void Derivate(Span<float> input, Span<float> output)
        {
            Debug.Assert(input != null);
            Debug.Assert(output != null);
            Debug.Assert(output.Length == input.Length);
            Debug.Assert(output.Length == Size);

            Intrinsics.SubstractScalar(1f, input, output); // 1 - target
            Intrinsics.Multiply(input, output, output); // output = target * (1-target)

           // for (int j = 0; j < Size; j++)
           // {
           //     output[j] = ActivationFunctions.SigmoidDerivative(input[j]);
           // }
        }


        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            Debug.Assert(target.Length == gamma.Length);
            Debug.Assert(target.Length == delta.Length);
            Debug.Assert(target.Length == Size);

            // gamma == difference times  activationDerivative(neuron value)

            //Span<float> values = stackalloc float[Size];

            Intrinsics.SubstractScalar(1f, target, gamma); // 1 - target
            Intrinsics.Multiply(target, gamma, gamma); // target * (1-target)
            Intrinsics.Multiply(delta, gamma, gamma); // gamma = delta * (target * (1-target))

            //for (int i = 0; i < Size; i++)
            //{
                //gamma[i] = delta[i] * (target[i] * (1f - target[i]));
            //};
        }
    }
}
