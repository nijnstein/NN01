using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    static public class MathExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(this float f)
        {
            f = (float)Math.Exp(f);
            return f / (1.0f + f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidDerivative(this float f)
        {
            return f * (1f - f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Swish(this float f)
        {
            return f * Sigmoid(f); 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SwishDerivative(this float f)      
        { 
            // see
            // https://sefiks.com/2018/08/21/swish-as-neural-networks-activation-function/
            float e = 1f + (float)Math.Exp(-f);
            return ((f * e) + (e - f)) / (e * e); 
        }


        public static float Max(this ReadOnlySpan<float> a)
        {
            float max = float.MinValue;
            unchecked
            {
                int i = 0;
                while (i < (a.Length & ~3))
                {
                    max = (a[i] > max ? a[i] : max);
                    max = (a[i + 1] > max ? a[i + 1] : max);
                    max = (a[i + 2] > max ? a[i + 2] : max);
                    max = (a[i + 3] > max ? a[i + 3] : max);
                    i += 4;
                }
                while (i < a.Length)
                {
                    max = (a[i] > max ? a[i] : max);
                    i++;
                }
            }
            return max;
        }


        public static void Softmax(ReadOnlySpan<float> input, Span<float> output, bool stable = false)
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
                    float ex = (float)Math.Exp(input[i]);
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
        public static void SoftmaxStable(ReadOnlySpan<float> input, Span<float> output)
        {
            float summed = 0;
            float max = Max(input);

            for (int i = 0; i < input.Length; i++)
            {
                float ex = (float)Math.Exp(input[i]) - max;
                summed += ex;
                output[i] = ex;
            }
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = output[i] / summed;
            }
        }
    }
}
