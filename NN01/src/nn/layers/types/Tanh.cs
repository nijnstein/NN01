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

namespace NN01
{
    public class TanhLayer : Layer
    {
        public override LayerActivationFunction ActivationType => LayerActivationFunction.Tanh;
        public TanhLayer(int size, int previousSize, Distribution weightInit = Distribution.Default, Distribution biasInit = Distribution.Default, bool skipInit = false)
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
            for (int j = 0; j < Size; j++)
            {
                float value = Intrinsics.SumWeighted(Weights[j], previous.Neurons) + Biases[j];

                if (Avx.IsSupported)
                {
                    Neurons[j] = value;
                }
                else
                {
                    Neurons[j] = value / (1f + MathF.Abs(value));
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
                    Span<Vector256<float>> n = MemoryMarshal.Cast<float, Vector256<float>>(Neurons);
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
                    Neurons[i] = Neurons[i] / (1f + MathF.Abs(Neurons[i]));
                    i++; 
                }

            }
        }
        public override void Derivate(Span<float> output)
        {
            Debug.Assert(output != null);
            Debug.Assert(output.Length == this.Neurons.Length);

            for (int j = 0; j < Size; j++)
            {
                output[j] = ActivationFunctions.TanhFastDerivative(Neurons[j]);
            }
        }

        public override void CalculateGamma(float[] delta, float[] gamma, float[] target)
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
