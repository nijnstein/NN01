using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Diagnostics.Contracts;

namespace NN01
{
    static public partial class Intrinsics
    {           /// <summary>
                /// => make sure data is aligned 
                /// a = Math.Abs(a)
                /// return a;
                /// </summary>
        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Abs(Span<float> a)
        {
            return Intrinsics.Abs(a, a);
        }

        /// <summary>
        /// => make sure data is aligned 
        /// output = Abs(a)
        /// </summary>
        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Abs(Span<float> a, Span<float> output)
        {
            int i = 0;

            if (Avx.IsSupported)
            {
                if (a.Length > 7)
                {
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> d = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    Vector256<float> sign = Vector256.Create(-0f); 
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~7))
                        {
                           // d[j] = //Abs(aa[j]);
                            d[j] = Avx.And(aa[j], sign);
                            i += 8;
                            j++;
                        }
                    }
                }
            }

            unchecked
            {
                while (i < a.Length)
                {
                    output[i] = MathF.Abs(a[i]);
                    i++;
                }
            }

            return output;
        }

        /// <summary>
        /// output = Abs(a)
        /// </summary>
        [Pure]
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> Abs(Vector256<float> a)
        {
            if (Avx2.IsSupported)
            {
                // dont create data by using shifts, in a loop we should and with sign
                return
                    Avx2.ShiftRightLogical(
                        Avx2.ShiftLeftLogical(a.AsInt32(), 1), 1).AsSingle(); ;
            }
            else
            {
                Span<float> f = stackalloc float[8];
                for (int i = 0; i < 8; i++)
                {
                    f[i] = MathF.Abs(a.GetElement(i));
                }
                return MemoryMarshal.Cast<float, Vector256<float>>(f)[0];
            }
        }          
    }
}
