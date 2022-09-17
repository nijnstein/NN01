using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using ILGPU.Runtime;
using NSS;

namespace NN01
{
    public class SoftmaxLayer : Layer
    {
        public SoftmaxLayer(int size) : base(size, size)
        {
        }

        public override LayerType ActivationType => LayerType.Softmax;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
           /* float LogSigmoid(float x)
            {
                if (x < -45f)
                    return 0f;
                else if (x > 45f)
                    return 1f;
                else
                    return 1f / (1f + MathF.Exp(-x));
            }

            for(int i = 0; i < inputData.Length; i++) 
            {
                outputData[i] = LogSigmoid(inputData[i]); 
            }
             */

             MathEx.Softmax(inputData, outputData);
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            MathEx.SoftmaxDerivative(input, output);
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            Intrinsics.Add(delta, target, gamma);
            
            for (int i = 0; i < gamma.Length; i++)
            {
                gamma[i] = gamma[i] * (1f - gamma[i]);
            }
           // Intrinsics.Substract(target, gamma, gamma);


            //MathEx.SoftmaxDerivative(target, gamma);
            //Intrinsics.Multiply(delta, gamma, gamma);
        }
    }
}
