using ILGPU.IR.Values;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static partial class MathEx
    {

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<float> a) => Variance(a, Average(a));

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<int> a) => Variance(a, Average(a));

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this float[][] a, float average)
        {
            int c = 0;
            float sq = 0;
            unchecked
            {
                for (int i = 0; i < a.Length; i++)
                {
                    sq += Intrinsics.SumSquaredDifferences(a[i], average);
                    c += a.Length;
                }
            }
            return sq / c;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<int> a, float average)
            => MathEx.SumSquaredDifferences(a, average) / a.Length;

        /// <summary>
        /// V =  sum((a - mean)^2) / a.length
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average) / a.Length;

        /// <summary>
        /// V =  sum((a - mean)^2)  /  sum(a)
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float VarianceN(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average) / Intrinsics.Sum(a);

        /// <summary>
        /// assumes all elements of a sum to a total of 1, variance is then just the summed squared differences
        /// </summary>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Variance1(this Span<float> a, float average) => Intrinsics.SumSquaredDifferences(a, average);


    }
}
