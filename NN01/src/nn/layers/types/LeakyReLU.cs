using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public class LeakyReLuLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.LeakyReLU;
        public LeakyReLuLayer(int size, int previousSize, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool skipInit = false, IRandom random = null)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializationType.Default ? LayerInitializationType.HeNormal : weightInit,
                  biasInit == LayerInitializationType.Default ? LayerInitializationType.dot01 : biasInit,
                  skipInit,
                  random
            )
        {
        }
        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span<float> values = stackalloc float[previous.Size];

            for (int j = 0; j < Size; j++)
            {
                // compute sum of weights multiplied with input neurons then add bias
                float value = Intrinsics.Sum(Intrinsics.Multiply(Weights[j], inputData, values)) + Biases[j];

                // leaky relu 
                outputData[j] = value < 0 ? 0.01f : value;
            }
        }

        public override void ReversedActivation(Layer next)
        {
            throw new NotImplementedException();
        }


        public override void Derivate(Span<float> output)
        {
            Debug.Assert(output != null);
            Debug.Assert(output.Length == this.Neurons.Length);

            for (int j = 0; j < Size; j++)
            {
                output[j] = ActivationFunctions.LeakyReLUDerivative(Neurons[j]);
            }
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            int i = 0, j = 0;

            if(Avx.IsSupported)
            {
                Vector256<float> a00 = Vector256<float>.Zero; 
                Vector256<float> a01 = Vector256.Create(0.01f);
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(delta);
                Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma);
                Span<Vector256<float>> t = MemoryMarshal.Cast<float, Vector256<float>>(target);

                while (i < (Size & ~7))
                {
                    // g = delta * (target < 0 ? 0.01 : 1)
                    g[j] = Avx.Multiply(d[j], Avx.BlendVariable(a01, a10, Avx.CompareLessThan(t[j], a00)));
                    i += 8;
                    j++; 
                }
            }
            while(i < Size)
            {  
                gamma[i] = delta[i] * (target[i] < 0 ? 0.01f : 1f);
                i++; 
            }
        }
    }

}
