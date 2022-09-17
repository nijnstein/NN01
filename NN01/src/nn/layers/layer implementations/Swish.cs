using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ILGPU.Runtime;
using NSS;

namespace NN01
{
    public class SwishLayer : ParameterLayer
    {
        public override LayerType ActivationType => LayerType.Swish;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;
        public override LayerInitializationType WeightInitializer => LayerInitializationType.HeNormal;
        public override LayerInitializationType BiasInitializer => LayerInitializationType.dot01;
        public SwishLayer(int size, int previousSize, bool skipInit = false, IRandom random = null)
            : base(size, previousSize, skipInit, random)
        {
        }

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span<float> buffer = stackalloc float[previous.Size];
            Span2D<float> w = Weights.AsSpan2D<float>();

            int j;  
            for (j = 0; j < Size; j++)
            {
                // compute sum of weights multiplied with input neurons then add bias
                float value = Intrinsics.SumWeighted(w.Row(j), inputData) + Biases[j];
                outputData[j] = value; // .Swish();
            }

            int i = 0; j = 0;
            if (Avx2.IsSupported)
            {
                Span<Vector256<float>> n = MemoryMarshal.Cast<float, Vector256<float>>(outputData);
                Vector256<float> a1 = Vector256.Create(1f);
                Vector256<float> a05 = Vector256.Create(0.5f);
                Vector256<float> sign = Vector256.Create(-0.0f); 

                while (i < (Size & ~7))
                {
                    // alternative sigmoid: 
                    Vector256<float> sigmoid; 
                    // const true -> use tanh sigmoid, == equal but much faster to calculate 
                    if (true)
                    {
                        // f *= 0.5f;
                        // return 0.5f + 0.5f * (f / (1f + MathF.Abs(f)));
                        sigmoid = Avx.Multiply(a05, n[j]);
                        //sigmoid = Intrinsics.MultiplyAdd(a05, Avx.Divide(sigmoid, Avx.Add(a1, Intrinsics.Abs(sigmoid))), a05); 
                        sigmoid = Intrinsics.MultiplyAdd(a05, Avx.Divide(sigmoid, Avx.Add(a1, Avx.And(sigmoid, sign))), a05); 

                    }
                    else
                    {
                        // v = exp(v)
                        sigmoid = Intrinsics.ExpAvx2Fast(n[j]);  // an inprecise exp would not be bad here 

                        // f = f / (1 + f)
                        sigmoid = Avx.Divide(sigmoid, Avx.Add(sigmoid, a1));
                    }
                    // swish = f * sigmoid
                    n[j] = Avx.Multiply(n[j], sigmoid);

                    i += 8;
                    j++;
                }
            }

            while(i < Size)
            {
                outputData[i] = outputData[i].Swish(); 
                i++;
            }
        }

        public override void Derivate(Span<float> input, Span<float> output)
        {
            Debug.Assert(input != null);
            Debug.Assert(output != null);
            Debug.Assert(output.Length == input.Length);

            for (int j = 0; j < Size; j++)
            {
                output[j] = ActivationFunctions.SwishDerivative(input[j]);
            }
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            int i = 0, j = 0;

            if (Avx.IsSupported)
            {
                Vector256<float> signChangeMask = Vector256.Create(-0.0f);
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(delta);
                Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma);
                Span<Vector256<float>> t = MemoryMarshal.Cast<float, Vector256<float>>(target);

                while (i < (Size & ~7))
                {
                    // e = 1 + exp(-f)
                    Vector256<float> e = Avx.Add(a10, Intrinsics.ExpAvx2Fast(Avx.Xor(t[j], signChangeMask)));

                    // derivate = (f * e + (e - f)) / (e * e);
                    Vector256<float> derivate = Avx.Divide(Avx.Add(Avx.Multiply(t[j], e), Avx.Subtract(e, t[j])), Avx.Multiply(e, e));

                    // g = delta * derivate
                    g[j] = Avx.Multiply(d[j], derivate); 

                    i += 8;
                    j++;
                }
            }
            while (i < Size)
            {
                gamma[i] = delta[i] * target[i].SwishDerivative();
                i++;
            }
        }
    }
}
