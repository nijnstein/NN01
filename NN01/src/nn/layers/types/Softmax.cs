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
 /*        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            MathEx.Softmax(inputData, outputData); 
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            MathEx.SoftmaxDerivative(input, output);
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            MathEx.SoftmaxDerivative(target, gamma);
            Intrinsics.Multiply(delta, gamma, gamma); 
        }

        public override void ReversedActivation(Layer next)
        {
            throw new NotImplementedException(); 
        }
   */ 
}
