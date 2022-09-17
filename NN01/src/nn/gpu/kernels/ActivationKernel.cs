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
      

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> GetGPU(LayerType activationFunction)
        {
                switch (activationFunction)
                {
                    case LayerType.ReLU: return ReLUGPUKernel;
                }
            return null;
        }

        public static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int> GetDerivateGPU(LayerType activationFunction)
        {
            switch (activationFunction)
                {
                    case LayerType.ReLU: return ReLUDerivateGPUKernel;
                }
            return null;
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
