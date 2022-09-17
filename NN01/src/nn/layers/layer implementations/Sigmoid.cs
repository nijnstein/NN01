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
    public class SigmoidLayer : ParameterLayer
    {
        public override LayerType ActivationType => LayerType.Sigmoid;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;
        public override LayerInitializationType WeightInitializer => LayerInitializationType.Glorot;
        public override LayerInitializationType BiasInitializer => LayerInitializationType.dot01;
        public SigmoidLayer(int size, int previousSize, bool skipInit = false, IRandom random = null)
            : base(size, previousSize, skipInit, random)
        {
        }
        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span2D<float> w = Weights.AsSpan2D<float>();
            Span<float> values = stackalloc float[Size];

            unchecked
            {
                for (int j = 0; j < Size; j++)
                {
                    // weighted sum of (w[j][k] * n[i][k])
                    // apply bias
                    values[j] = Intrinsics.SumWeighted(w.Row(j), inputData);
                }
            }
            Intrinsics.Add(values, Biases, values); 
            Intrinsics.Exp(values, values);
            Intrinsics.AddScalar(values, 1, outputData);
            Intrinsics.Divide(values, outputData, outputData);
            // for (int j = 0; j < Size; j++)
            // {
            //     // sigmoid activation
            //     float f = values2[j];
            //     outputData[j] = f / (1.0f + f);
            // }            
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            Debug.Assert(input != null);
            Debug.Assert(output != null);
            Debug.Assert(output.Length == input.Length);
            Debug.Assert(output.Length == Size);

            Intrinsics.SubstractScalar(1f, input, output); // 1 - target
            Intrinsics.Multiply(input, output, output); // output = target * (1-target)
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            Debug.Assert(target.Length == gamma.Length);
            Debug.Assert(target.Length == delta.Length);
            Debug.Assert(target.Length == Size);

            // gamma == difference times  activationDerivative(neuron value)
            Intrinsics.SubstractScalar(1f, target, gamma); // 1 - target
            Intrinsics.Multiply(target, gamma, gamma); // target * (1-target)
            Intrinsics.Multiply(delta, gamma, gamma); // gamma = delta * (target * (1-target))
        }
    }
}
