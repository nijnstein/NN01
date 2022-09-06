using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NSS
{
    public static partial class Intrinsics
    {
        static Vector256<float> v256_one = Vector256.Create(1f);
        static Vector256<float> v256_05 = Vector256.Create(0.5f);
        static Vector256<float> v256_minfloat = Vector256.Create(0x00800000).AsSingle();
        static Vector256<int> v256_0x7f = Vector256.Create((int)0x7f);

        static Vector256<float> v256_mant_mask = Vector256.Create(0x7f800000).AsSingle();
        static Vector256<float> v256_inv_mant_mask = Vector256.Create(~0x7f800000).AsSingle();
        static Vector256<float> v256_sign_mask = Vector256.Create(0x80000000).AsSingle();
        static Vector256<float> v256_inv_sign_mask = Vector256.Create(~0x80000000).AsSingle();

        static Vector256<float> v256_cephes_SQRTHF = Vector256.Create(0.707106781186547524f);
        static Vector256<float> v256_cephes_log_p0 = Vector256.Create(7.0376836292E-2f);
        static Vector256<float> v256_cephes_log_p1 = Vector256.Create(-1.1514610310E-1f);
        static Vector256<float> v256_cephes_log_p2 = Vector256.Create(1.1676998740E-1f);
        static Vector256<float> v256_cephes_log_p3 = Vector256.Create(-1.2420140846E-1f);
        static Vector256<float> v256_cephes_log_p4 = Vector256.Create(+1.4249322787E-1f);
        static Vector256<float> v256_cephes_log_p5 = Vector256.Create(-1.6668057665E-1f);
        static Vector256<float> v256_cephes_log_p6 = Vector256.Create(+2.0000714765E-1f);
        static Vector256<float> v256_cephes_log_p7 = Vector256.Create(-2.4999993993E-1f);
        static Vector256<float> v256_cephes_log_p8 = Vector256.Create(+3.3333331174E-1f);
        static Vector256<float> v256_cephes_log_q1 = Vector256.Create(-2.12194440e-4f);
        static Vector256<float> v256_cephes_log_q2 = Vector256.Create(0.693359375f);

        // return NaN for x <= 0
        //
        // converted into .net from: https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h
        //
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> LogAvx2(Vector256<float> x)
        {
            Vector256<int> imm0;
            Vector256<float> one = v256_one;

            //v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
            Vector256<float> invalid_mask = Avx.Compare(x, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanOrEqualSignaling);

            x = Avx.Max(x, v256_minfloat);  /* cut off denormalized stuff */

            // can be done with AVX2 (avx uses 2 lanes) 
            imm0 = Avx2.ShiftRightLogical(x.AsInt32(), 23);

            /* keep only the fractional part */
            x = Avx2.And(x, v256_inv_mant_mask);
            x = Avx2.Or(x, v256_05);

            // this is again another AVX2 instruction
            imm0 = Avx2.Subtract(imm0, v256_0x7f);

            Vector256<float> e = imm0.AsSingle();

            e = Avx.Add(e, one);

            /* part2: 
               if( x < SQRTHF ) {
                 e -= 1;
                 x = x + x - 1.0;
               } else { x = x - 1.0; }
            */
            //v8sf mask = _mm256_cmplt_ps(x, *(v8sf*)_ps256_cephes_SQRTHF);
            Vector256<float> mask = Avx.Compare(x, v256_cephes_SQRTHF, FloatComparisonMode.OrderedLessThanSignaling);
            Vector256<float> tmp = Avx.And(x, mask);

            x = Avx.Subtract(x, one);
            e = Avx.Subtract(e, Avx.And(one, mask));
            x = Avx.And(x, tmp);

            Vector256<float> z = Avx.Multiply(x, x);
            Vector256<float> y = v256_cephes_log_p0;

            // curve fitting.. 
            if (Fma.IsSupported)
            {
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p1);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p2);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p3);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p4);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p5);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p6);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p7);
                y = Fma.MultiplyAdd(y, x, v256_cephes_log_p8);
                y = Avx.Multiply(y, x);
            }
            else
            {
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p1);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p2);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p3);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p4);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p5);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p6);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p7);
                y = Avx.Multiply(y, x);
                y = Avx.Add(y, v256_cephes_log_p8);
                y = Avx.Multiply(y, x);
            } 

            y = Avx.Multiply(y, z);
            tmp = Avx.Multiply(e, v256_cephes_log_q1);
            y = Avx.Add(y, tmp);
            tmp = Avx.Multiply(z, v256_05);
            y = Avx.Subtract(y, tmp);
            tmp = Avx.Multiply(e, v256_cephes_log_q2);
            x = Avx.Add(x, y);
            x = Avx.Add(x, tmp);
            x = Avx.Or(x, invalid_mask); // negative arg will be NAN
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Log(ReadOnlySpan<float> a, Span<float> output)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Log(MemoryMarshal.CreateSpan<float>(ref reff, a.Length), output);
        }

        /// <summary>
        /// a = Log(a); 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Log(Span<float> a)
        {
            return Log(a, a); 
        }

        /// <summary>
        /// output = Log(a)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Log(Span<float> a, Span<float> output)
        {
            int i = 0;
            if (Avx2.IsSupported)
            {
                if (a.Length > 7)
                {         
                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> o = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~7))
                        {
                            o[j] = LogAvx2(aa[j]);
                            i += 8;
                            j++;
                        }
                    }
                }
            }
            unchecked
            {
                if (i < (a.Length & ~3))
                {
                    while (i < (a.Length & ~3))
                    {
                        output[i + 0] = MathF.Log(a[i + 0]);
                        output[i + 1] = MathF.Log(a[i + 1]);
                        output[i + 2] = MathF.Log(a[i + 2]);
                        output[i + 3] = MathF.Log(a[i + 3]);
                        i += 4;
                    }
                }
                while (i < a.Length)
                {
                    output[i] = MathF.Log(a[i]);
                    i++;
                }
            }
            return output;
        }

    }
}
