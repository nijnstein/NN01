using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static partial class Intrinsics
    {
        // if true 1 order higher precision (1 more coeff. upto ()e-7
        // otherwise 3.1575e-6 (USE_FMA = 0); 3.1533e-6 (USE_FMA = 1)
        const bool HighPrecision = true;

        /// <summary>
        /// compute exp(x) for x in [-87.33654f, 88.72283]  
        /// from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> ExpAvx2(Vector256<float> x)
        {
            Vector256<float> t, f, p, r;
            Vector256<int> i, j;

            Vector256<float> l2e = Vector256.Create(1.442695041f);      /* log2(e) */
            Vector256<float> l2h = Vector256.Create(-6.93145752e-1f);   /* -log(2)_hi */
            Vector256<float> l2l = Vector256.Create(-1.42860677e-6f);   /* -log(2)_lo */

            /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
            /* maximum relative error: 1.7428e-7 (USE_FMA = 0); 1.6586e-7 (USE_FMA = 1) */
            Vector256<float> c0 = Vector256.Create(0.008301110f);
            Vector256<float> c1 = Vector256.Create(0.041906696f);
            Vector256<float> c2 = Vector256.Create(0.166674897f);
            Vector256<float> c3 = Vector256.Create(0.499990642f);
            Vector256<float> c4 = Vector256.Create(0.999999762f);
            Vector256<float> c5 = Vector256.Create(1.000000000f);

            /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
            t = Avx.Multiply(x, l2e);         /* t = log2(e) * x */
            r = Avx.RoundToNearestInteger(t); /* r = rint (t) */

            if (Fma.IsSupported)
            {
                f = Fma.MultiplyAdd(r, l2h, x); /* x - log(2)_hi * r */
                f = Fma.MultiplyAdd(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
            }
            else
            {
                p = Avx.Multiply(r, l2h); /* log(2)_hi * r */
                f = Avx.Add(x, p);        /* x - log(2)_hi * r */
                p = Avx.Multiply(r, l2l); /* log(2)_lo * r */
                f = Avx.Add(f, p);        /* f = x - log(2)_hi * r - log(2)_lo * r */
            }

            i = Avx2.ConvertToVector256Int32(t);    /* i = (int)rint(t) */

            /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
            p = c0;                          /* c0 */

            if (Fma.IsSupported)
            {
                p = Fma.MultiplyAdd(p, f, c1);  /* c0*f+c1 */
                p = Fma.MultiplyAdd(p, f, c2);  /* (c0*f+c1)*f+c2 */
                p = Fma.MultiplyAdd(p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
                p = Fma.MultiplyAdd(p, f, c4);
                if (HighPrecision)
                {
                    p = Fma.MultiplyAdd(p, f, c5);  /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                }
            }
            else
            {
                p = Avx.Multiply(p, f);   /* c0*f */
                p = Avx.Add(p, c1);       /* c0*f+c1 */
                p = Avx.Multiply(p, f);   /* (c0*f+c1)*f */
                p = Avx.Add(p, c2);       /* (c0*f+c1)*f+c2 */
                p = Avx.Multiply(p, f);   /* ((c0*f+c1)*f+c2)*f */
                p = Avx.Add(p, c3);       /* ((c0*f+c1)*f+c2)*f+c3 */
                p = Avx.Multiply(p, f);   /* (((c0*f+c1)*f+c2)*f+c3)*f */
                p = Avx.Add(p, c4);       /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                if (HighPrecision)
                {
                    p = Avx.Multiply(p, f);        /* (((c0*f+c1)*f+c2)*f+c3)*f */
                    p = Avx.Add(p, c5);       /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                }
            }
            /* exp(x) = 2^i * p */
            j = Avx2.ShiftLeftLogical(i, 23); /* i << 23 */
            r = Avx2.Add(j, p.AsInt32()).AsSingle();  // r = p * 2^i 

            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> ExpAvx2Fast(Vector256<float> x)
        {
            Vector256<float> t, f, p, r;
            Vector256<int> i, j;

            Vector256<float> l2e = Vector256.Create(1.442695041f);      /* log2(e) */
            Vector256<float> l2h = Vector256.Create(-6.93145752e-1f);   /* -log(2)_hi */
            Vector256<float> l2l = Vector256.Create(-1.42860677e-6f);   /* -log(2)_lo */

            /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
            Vector256<float> c0 = Vector256.Create(0.041944388f);
            Vector256<float> c1 = Vector256.Create(0.168006673f);
            Vector256<float> c2 = Vector256.Create(0.499999940f);
            Vector256<float> c3 = Vector256.Create(0.999956906f);

            /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
            t = Avx.Multiply(x, l2e);         /* t = log2(e) * x */
            r = Avx.RoundToNearestInteger(t); /* r = rint (t) */

            if (Fma.IsSupported)
            {
                f = Fma.MultiplyAdd(r, l2h, x); /* x - log(2)_hi * r */
                f = Fma.MultiplyAdd(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
            }
            else
            {
                p = Avx.Multiply(r, l2h); /* log(2)_hi * r */
                f = Avx.Add(x, p);        /* x - log(2)_hi * r */
                p = Avx.Multiply(r, l2l); /* log(2)_lo * r */
                f = Avx.Add(f, p);        /* f = x - log(2)_hi * r - log(2)_lo * r */
            }

            i = Avx2.ConvertToVector256Int32(t);    /* i = (int)rint(t) */

            /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
            p = c0;                          /* c0 */

            if (Fma.IsSupported)
            {
                p = Fma.MultiplyAdd(p, f, c1);  /* c0*f+c1 */
                p = Fma.MultiplyAdd(p, f, c2);  /* (c0*f+c1)*f+c2 */
                p = Fma.MultiplyAdd(p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
            }
            else
            {
                p = Avx.Multiply(p, f);   /* c0*f */
                p = Avx.Add(p, c1);       /* c0*f+c1 */
                p = Avx.Multiply(p, f);   /* (c0*f+c1)*f */
                p = Avx.Add(p, c2);       /* (c0*f+c1)*f+c2 */
                p = Avx.Multiply(p, f);   /* ((c0*f+c1)*f+c2)*f */
                p = Avx.Add(p, c3);       /* ((c0*f+c1)*f+c2)*f+c3 */
                p = Avx.Multiply(p, f);   /* (((c0*f+c1)*f+c2)*f+c3)*f */
            }
            /* exp(x) = 2^i * p */
            j = Avx2.ShiftLeftLogical(i, 23); /* i << 23 */
            r = Avx2.Add(j, p.AsInt32()).AsSingle();  // r = p * 2^i 

            return r;
        }



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Exp(ReadOnlySpan<float> a, Span<float> output)
        {
            float reff = MemoryMarshal.GetReference(a);
            return Exp(MemoryMarshal.CreateSpan<float>(ref reff, a.Length), output);
        }

        /// <summary>
        /// a = Exp(a); 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> Exp(Span<float> a)
        {
            return Exp(a, a); 
        }

        /// <summary>
        /// output = Exp(a)
        /// </summary>
        public static Span<float> Exp(Span<float> a, Span<float> output)
        {
            int i = 0;
            if (Avx2.IsSupported)
            {
                if (a.Length > 7)
                {
                    Vector256<float> t, f, p, r;
                    Vector256<int> ii;

                    Vector256<float> l2e = Vector256.Create(1.442695041f); /* log2(e) */
                    Vector256<float> l2h = Vector256.Create(-6.93145752e-1f); /* -log(2)_hi */
                    Vector256<float> l2l = Vector256.Create(-1.42860677e-6f); /* -log(2)_lo */

                    /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
                    /* maximum relative error: 1.7428e-7 (USE_FMA = 0); 1.6586e-7 (USE_FMA = 1) */
                    Vector256<float> c0 = Vector256.Create(0.008301110f);
                    Vector256<float> c1 = Vector256.Create(0.041906696f);
                    Vector256<float> c2 = Vector256.Create(0.166674897f);
                    Vector256<float> c3 = Vector256.Create(0.499990642f);
                    Vector256<float> c4 = Vector256.Create(0.999999762f);
                    Vector256<float> c5 = Vector256.Create(1.000000000f);

                    Span<Vector256<float>> aa = MemoryMarshal.Cast<float, Vector256<float>>(a);
                    Span<Vector256<float>> o = MemoryMarshal.Cast<float, Vector256<float>>(output);
                    unchecked
                    {
                        int j = 0;
                        while (i < (a.Length & ~7))
                        {
                            /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
                            t = Avx.Multiply(aa[j], l2e);           /* t = log2(e) * x */
                            r = Avx.RoundToNearestInteger(t);       /* r = rint (t) */

                            if (Fma.IsSupported)
                            {
                                f = Fma.MultiplyAdd(r, l2h, aa[j]); /* x - log(2)_hi * r */
                                f = Fma.MultiplyAdd(r, l2l, f);     /* f = x - log(2)_hi * r - log(2)_lo * r */
                            }
                            else
                            {
                                p = Avx.Multiply(r, l2h);           /* log(2)_hi * r */
                                f = Avx.Add(aa[j], p);              /* x - log(2)_hi * r */
                                p = Avx.Multiply(r, l2l);           /* log(2)_lo * r */
                                f = Avx.Add(f, p);                  /* f = x - log(2)_hi * r - log(2)_lo * r */
                            }

                            ii = Avx2.ConvertToVector256Int32(t);   /* i = (int)rint(t) */
                                                                    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
                            p = c0;                                 /* c0 */
                            if (Fma.IsSupported)
                            {
                                p = Fma.MultiplyAdd(p, f, c1);      /* c0*f+c1 */
                                p = Fma.MultiplyAdd(p, f, c2);      /* (c0*f+c1)*f+c2 */
                                p = Fma.MultiplyAdd(p, f, c3);      /* ((c0*f+c1)*f+c2)*f+c3 */
                                p = Fma.MultiplyAdd(p, f, c4);
                                p = Fma.MultiplyAdd(p, f, c5);      /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                            }
                            else
                            {
                                p = Avx.Multiply(p, f);             /* c0*f */
                                p = Avx.Add(p, c1);                 /* c0*f+c1 */
                                p = Avx.Multiply(p, f);             /* (c0*f+c1)*f */
                                p = Avx.Add(p, c2);                 /* (c0*f+c1)*f+c2 */
                                p = Avx.Multiply(p, f);             /* ((c0*f+c1)*f+c2)*f */
                                p = Avx.Add(p, c3);                 /* ((c0*f+c1)*f+c2)*f+c3 */
                                p = Avx.Multiply(p, f);             /* (((c0*f+c1)*f+c2)*f+c3)*f */
                                p = Avx.Add(p, c4);                 /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                                p = Avx.Multiply(p, f);             /* (((c0*f+c1)*f+c2)*f+c3)*f */
                                p = Avx.Add(p, c5);                 /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
                            }

                            /* exp(x) = 2^i * p */
                            o[j] = Avx2.Add(Avx2.ShiftLeftLogical(ii, 23), p.AsInt32()).AsSingle();
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
                        output[i + 0] = MathF.Exp(a[i + 0]);
                        output[i + 1] = MathF.Exp(a[i + 1]);
                        output[i + 2] = MathF.Exp(a[i + 2]);
                        output[i + 3] = MathF.Exp(a[i + 3]);
                        i += 4;
                    }
                }
                while (i < a.Length)
                {
                    output[i] = MathF.Exp(a[i]);
                    i++;
                }
            }
            return output;
        }

    }
}
