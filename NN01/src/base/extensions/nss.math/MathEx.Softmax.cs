using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static partial class MathEx
    {
        public static void Softmax(this Span<float> input, Span<float> output, bool stable = false)
        {
            // softmax == each exponent of every component divided by the sum of of all exponents 
            if (stable)
            {
                // stable done in other function,
                // - dont want inner loop branches
                // - dont want to modify input 
                // - dont want unneccessary memory operations on ALL non stable stofmax calls
                SoftmaxStable(input, output);
            }
            else
            {
                float summed = 0;
                for (int i = 0; i < input.Length; i++)
                {
                    float ex = MathF.Exp(input[i]);
                    summed += ex;
                    output[i] = ex;
                }
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = output[i] / summed;
                }
            }
        }

        /// <summary>
        /// stabelized softmax, max(input) is used to pull data into the negative in a reduced spread allowing exponentiation towards zero instead of infinity on large input values
        /// </summary>
        public static void SoftmaxStable(Span<float> input, Span<float> output)
        {
            float summed = 0;
            float max = Max(input);

            for (int i = 0; i < input.Length; i++)
            {
                float ex = MathF.Exp(input[i]) - max;
                summed += ex;
                output[i] = ex;
            }
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = output[i] / summed;
            }
        }

        public static void SoftmaxDerivative(Span<float> activation, Span<float> gradient)
        {
            Debug.Assert(activation.Length == gradient.Length);

            for (int i = 0; i < activation.Length; i++)
            {
             /* for (int j = 0; j < actual.Length; i++)
                {
                    if (i == j)
                    {*/
                        gradient[i] = activation[i] * (1 - activation[i]); 
                /*    }
                    else
                    {
                        gradient[i] = -activation[i] * actual[i];
                    }
                } */ 
            }
        }



        /// <summary>
        /// softplus  = log(exp(x) + 1) 
        /// </summary>
        public static void Softplus(this Span<float> input, Span<float> output)
        {
            Intrinsics.Exp(input, output);
            Intrinsics.AddScalar(output, 1f, output); 
            Intrinsics.Log(output, output);
        }

    }
}
