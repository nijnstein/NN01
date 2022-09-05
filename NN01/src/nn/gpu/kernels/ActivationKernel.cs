using ILGPU;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.IR;
using ILGPU.IR.Types;
using ILGPU.Runtime;
using NSS;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace NN01
{
    public static class ActivationKernel
    {
        public static Action<NetworkBuffers<float>, NetworkBuffers<float>> GetCPU(LayerActivationFunction activationFunction, bool derivate)
        {
            if (derivate)
            {
                switch (activationFunction)
                {
               //     case LayerActivationFunction.ReLU: return ReLuDerivateCPUKernel;
                }
            }
            else
            {
                switch (activationFunction)
                {
                    case LayerActivationFunction.ReLU: return ReLUCPUKernel; 
                }
            }
            Debug.Assert(false, $"no cpu kernel found for {(derivate ? "derivate of " : "")}{activationFunction}"); 
            return null;
        }
        

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> GetGPU(LayerActivationFunction activationFunction)
        {
                switch (activationFunction)
                {
                    case LayerActivationFunction.ReLU: return ReLUGPUKernel;
                }
            return null;
        }

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int> GetDerivateGPU(LayerActivationFunction activationFunction)
        {
            switch (activationFunction)
                {
                    case LayerActivationFunction.ReLU: return ReLUDerivateGPUKernel;
                }
            return null;
        }



        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ReLUCPUKernel(NetworkBuffers<float> layer, NetworkBuffers<float> next)
        {
            var w01 = layer.Weights.Span; 
            var b1 = layer.Bias.Span;
            var n0 = layer.Neurons.Span;
            var n1 = next.Neurons.Span;

            //
            //  weights [current layer] [next layer]    
            //
            //  stride = current layer size
            //
            //  n1[] =  sum (n0[] * w01[]) + b 
            //

            int i = 0;
            unchecked
            {
                for (int j = 0; j < n1.Length; j++)
                {
                    var ww = w01.Slice(i, n0.Length);
                    i += n0.Length;

                    // compute sum of weights multiplied with input neurons then add bias
                    float value = Intrinsics.SumWeighted(ww, n0) + b1[j];

                    // perform ReLU activation 
                    n1[j] = Math.Max(value, 0);
                }
            }
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ReLUDerivateCPUKernel(NetworkBuffers<float> layer, NetworkBuffers<float> next)
        {
            var w01 = layer.Weights.Span;
            var b1 = layer.Bias.Span;
            var n0 = layer.Neurons.Span;
            var n1 = next.Neurons.Span;


        }

        static public void ReLUGPUKernel(Index1D index, ArrayView<float> n0, ArrayView<float> n1, ArrayView<float> w01, ArrayView<float> b1)
        {
            // index 0...j foreach [] in n01 
            int stride = (int)n0.Length;
            int i1 = index * stride;
            float sum = 0f;

            // weighted sum 
            for (int i0 = 0; i0 < stride; i0++)
            {
                sum += w01[i0 + i1] * n0[i0];
            }

            sum += b1[index];

            // ReLU
            n1[index] = MathF.Max(sum, 0);
        }

        static public void ReLUDerivateGPUKernel(Index1D index, ArrayView<float> delta, ArrayView<float> gamma, ArrayView2D<float, Stride2D.DenseX> expectation, int sampleIndex)
        {
            gamma[index] = delta[index] * (expectation[sampleIndex, index] < 0 ? 0 : 1f);
        }
    }
}
