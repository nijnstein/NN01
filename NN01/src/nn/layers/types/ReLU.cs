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

namespace NN01
{
    public class ReLuLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.ReLU;
        public ReLuLayer(int size, int previousSize, Distribution weightInit = Distribution.Default, Distribution biasInit = Distribution.Default, bool skipInit = false)
            : base
            (
                  size, 
                  previousSize, 
                  weightInit == Distribution.Default ? Distribution.HeNormal : weightInit,
                  biasInit == Distribution.Default ? Distribution.Random : biasInit,
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

                // relu 
                Neurons[j] = Math.Max(value, 0);
            }
        }

        /// <summary>
        /// get derivate of current activation state 
        /// </summary>
        /// <param name="output">derivate output</param>
        public override void Derivate(Span<float> output)
        {
            Debug.Assert(output != null);
            Debug.Assert(output.Length == this.Neurons.Length);
                  
            for (int j = 0; j < Size; j++)
            {
                output[j] = ActivationFunctions.ReLUDerivative(Neurons[j]);
            }
        }

        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
        {
            int i = 0, j = 0;

            if (Avx.IsSupported)
            {
                Vector256<float> a00 = Vector256<float>.Zero;
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(delta);
                Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma);
                Span<Vector256<float>> t = MemoryMarshal.Cast<float, Vector256<float>>(target);

                while (i < (Size & ~7))
                {
                    // g = delta * (target < 0 ? 0.01 : 1)
                    g[j] = Avx.Multiply(d[j], Avx.BlendVariable(a00, a10, Avx.CompareLessThan(t[j], a00)));
                    i += 8;
                    j++;
                }
            }
            while (i < Size)
            {
                gamma[i] = delta[i] * (target[i] < 0 ? 0f : 1f);
                i++;
            }
        }
    }
}
