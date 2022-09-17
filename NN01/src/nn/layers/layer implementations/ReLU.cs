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
    public class ReLuLayer : ParameterLayer
    {
        public override LayerType ActivationType => LayerType.ReLU;
        public override LayerConnectedness Connectedness => LayerConnectedness.Full;
        public override LayerInitializationType WeightInitializer => LayerInitializationType.HeNormal;
        public override LayerInitializationType BiasInitializer => LayerInitializationType.dot01;

        public ReLuLayer(int size, int previousSize, bool skipInit = false, IRandom random = null)
            : base(size, previousSize, skipInit, random)
        { 
        }

        public override void Activate(Layer previous, Span<float> inputData, Span<float> outputData)
        {
            Span<float> values = stackalloc float[previous.Size];
            Span2D<float> w = Weights.AsSpan2D<float>();

            for (int j = 0; j < Size; j++)
            {
                // compute sum of weights multiplied with input neurons then add bias
                float value = Intrinsics.Sum(Intrinsics.Multiply(w.Row(j), inputData, values)) + Biases[j];

                // relu 
                outputData[j] = Math.Max(value, 0);
            }
        }

        /// <summary>
        /// get derivate of current activation state 
        /// </summary>
        /// <param name="output">derivate output</param>
        public override void Derivate(Span<float> input, Span<float> output)
        {
            Debug.Assert(output != null);
            Debug.Assert(input != null);
            Debug.Assert(output.Length == input.Length);

            //for (int j = 0; j < Size; j++)
            //{
            //    output[j] = ActivationFunctions.ReLUDerivative(input[j]);
            //}

            int i = 0, j = 0;

            if (Avx.IsSupported)
            {
                Vector256<float> a00 = Vector256<float>.Zero;
                Vector256<float> a10 = Vector256.Create(1f);
                Span<Vector256<float>> ii = MemoryMarshal.Cast<float, Vector256<float>>(input);
                Span<Vector256<float>> oo = MemoryMarshal.Cast<float, Vector256<float>>(output);

                while (i < (Size & ~7))
                {
                    // g = delta * (target < 0 ? 0.01 : 1)
                    oo[j] = Avx.BlendVariable(a00, a10, Avx.CompareLessThan(ii[j], a00));
                    i += 8;
                    j++;
                }
            }
            while (i < Size)
            {
                output[i] = input[i] < 0 ? 0f : 1f;
                i++;
            }
        }

        public override void CalculateGamma(Span<float> delta, Span<float> gamma, Span<float> target)
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
