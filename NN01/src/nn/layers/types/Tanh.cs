using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using NSS;

namespace NN01
{
    public class TanhLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Tanh;

        //
        //  for weights, use glorot, xavier etc.  but not a normal 
        // 
        //  tanh wants close to sum(all) == 1 
        //

        public TanhLayer(int size, int previousSize, LayerInitializationType weightInit = LayerInitializationType.Default, LayerInitializationType biasInit = LayerInitializationType.Default, bool softmax = false, bool skipInit = false, IRandom random = null)
            : base
            (
                  size,
                  previousSize,
                  weightInit == LayerInitializationType.Default ? LayerInitializationType.Xavier : weightInit,
                  biasInit == LayerInitializationType.Default ? LayerInitializationType.dot01 : biasInit,
                  softmax,
                  skipInit, 
                  random
            )
        {
        }
        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span2D<float> w = Weights.AsSpan2D<float>();

            for (int j = 0; j < Size; j++)
            {
                float value = Intrinsics.SumWeighted(w.Row(j), inputData) + Biases[j];
                if (Avx.IsSupported)
                {
                    outputData[j] = value;
                }
                else
                {
                    outputData[j] = value / (1f + MathF.Abs(value));
                }
                
            }

            //Neurons[j] = (float)Math.Tanh(value);
            if (Avx.IsSupported)
            {
                int i = 0;
                int j = 0;
                if (Size > 7)
                {
                    Vector256<float> a1 = Vector256.Create(1f); 
                    Span<Vector256<float>> n = MemoryMarshal.Cast<float, Vector256<float>>(outputData);
                    while (i < (Size & ~7))
                    {
                        n[j] = Avx.Divide(n[j] 
                            ,
                            Avx.Add(a1, 
                                Avx2.ShiftRightLogical(
                                    Avx2.ShiftLeftLogical(n[j].AsInt32(), 1) 
                                    , 1)
                                .AsSingle()));
                        j += 1;
                        i += 8;
                    }
                }
                while (i < Size)
                {
                    outputData[i] = outputData[i] / (1f + MathF.Abs(outputData[i]));
                    i++; 
                }

            }
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

            //for (int j = 0; j < Size; j++)
            //{
            //    output[j] = ActivationFunctions.TanhFastDerivative(input[j]);
            //}

            int i = 0, j = 0;
            if (Avx.IsSupported)
            {
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> ii = MemoryMarshal.Cast<float, Vector256<float>>(input);
                Span<Vector256<float>> oo = MemoryMarshal.Cast<float, Vector256<float>>(output);

                while (i < (Size & ~7))
                {
                    oo[j] = Avx.Subtract(a10, Avx.Multiply(ii[j], ii[j]));
                    i += 8;
                    j++;
                }
            }
            while (i < Size)
            {
                output[i] = (1 - input[i] * input[i]);
                i++;
            }
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
        {
            int i = 0, j = 0;

            if (Avx.IsSupported)
            {
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(delta);
                Span<Vector256<float>> g = MemoryMarshal.Cast<float, Vector256<float>>(gamma);
                Span<Vector256<float>> t = MemoryMarshal.Cast<float, Vector256<float>>(target);

                while (i < (Size & ~7))
                {
                    g[j] = Avx.Multiply(d[j], Avx.Subtract(a10, Avx.Multiply(t[j], t[j])));
                    i += 8;
                    j++;
                }
            }
            while (i < Size)
            {
                gamma[i] = delta[i] * (1 - target[i] * target[i]);
                i++;
            }
        }
    }
}
