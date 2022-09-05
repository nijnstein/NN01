using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;

namespace NSS
{
    /// <summary>
    /// FFT Based on: 
    /// http://local.wasp.uwa.edu.au/~pbourke/miscellaneous/dft/index.html
    /// by Paul Bourke
    /// </summary>
    static public class Fourier
    {
        /// <summary>
        /// Disrete Fourier Transform (the slow one) 
        /// </summary>
        /// <param name="n">number of samples</param>
        /// <param name="dir">1 = forward, -1 = inverse</param>
        /// <param name="gRe">input real part</param>
        /// <param name="gIm">input imaginary part</param>
        /// <param name="GRe">output real part</param>
        /// <param name="GIm">output imaginary part</param>
        static public void DFT(int n, int dir, float[] gRe, float[] gIm, float[] GRe, float[] GIm)
        {
            for (int w = 0; w < n; w++)
            {
                GRe[w] = GIm[w] = 0;
                for (int x = 0; x < n; x++)
                {
                    float a = -2 * MathF.PI * w * x / n;
                    if (dir < 0) a = -a;
                    float ca = MathF.Cos(a);
                    float sa = MathF.Sin(a);
                    GRe[w] += gRe[x] * ca - gIm[x] * sa;
                    GIm[w] += gRe[x] * sa + gIm[x] * ca;
                }
                if (dir > 0)
                {
                    GRe[w] /= n;
                    GIm[w] /= n;
                }
            }
        }

        /// <summary>
        /// Compute the 2d FFT from the complex input data
        /// </summary>
        /// <param name="real">the real part</param>
        /// <param name="complex">the complex part</param>
        /// <param name="width">the width of the 2d array of data, must be in a power of 2</param>
        /// <param name="height">the height of the 2d array of data, must be in a power of 2</param>
        /// <param name="dir">direction of the transform, 1 for forward, -1 for reverse (inversed)</param>
        /// <returns>returns true when successfull, false when the dimensions are not in 
        /// a power of 2 </returns>
        static public bool FFT(float[] real, float[] complex, uint width, uint height, int dir)
        {
            // both the width and height must be a power of 2
            uint m_width;
            uint m_height;

            if (!MathEx.PowerOf2(width, out m_width)) return false;
            if (!MathEx.PowerOf2(height, out m_height)) return false;

            // the 2d fourier transform is split up in 2 1D transforms 
            float[] real1D = new float[width];
            float[] complex1D = new float[width];

            // Apply the transform to the rows
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    real1D[x] = real[y * width + x];
                    complex1D[x] = complex[y * width + x];
                }

                FFT(dir, m_width, real1D, complex1D);

                for (int x = 0; x < width; x++)
                {
                    real[y * width + x] = real1D[x];
                    complex[y * width + x] = complex1D[x];
                }
            }

            // then to the columns
            real1D = new float[height];
            complex1D = new float[height];

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    real1D[y] = real[y * width + x];
                    complex1D[y] = complex[y * width + x];
                }

                FFT(dir, m_height, real1D, complex1D);

                for (int y = 0; y < height; y++)
                {
                    real[y * width + x] = real1D[y];
                    complex[y * width + x] = complex1D[y];
                }
            }

            return true;
        }


        /// <summary>
        /// This computes an in-place real to real FFT
        ///   data is an array of real samples of 2^m points.
        ///  
        ///   dir =  1 gives forward transform
        ///   dir = -1 gives reverse transform
        ///
        ///     Formula: forward
        ///                  N-1
        ///                  ---
        ///              1   \          - j k 2 pi n / N
        ///      X(n) = ---   >   x(k) e                    = forward transform
        ///              N   /                                n=0..N-1
        ///                  ---
        ///                  k=0
        ///
        ///      Formula: reverse
        ///                  N-1
        ///                  ---
        ///                  \          j k 2 pi n / N
        ///      X(n) =       >   x(k) e                    = forward transform
        ///                  /                                n=0..N-1
        ///                  ---
        ///                  k=0
        /// <summary>
        /// 1D Fast Fourier Transform
        /// </summary>
        /// <param name="dir"> 1 = forward, -1 = inverse</param>
        /// <param name="m">number of samples</param>
        /// <param name="real">the real part</param>
        /// <param name="complex">the complex part</param>
        /// <returns></returns>
        public static bool FFT(int dir, uint m, float[] real, float[] complex)
        {
            long nn, i, i1, j, k, i2, l, l1, l2;
            float c1, c2, tx, t1, t2, u1, u2, z;

            /* Calculate the number of points */
            nn = 1;
            for (i = 0; i < m; i++) nn *= 2;

            /* Do the bit reversal */
            i2 = nn >> 1;
            j = 0;

            for (i = 0; i < nn - 1; i++)
            {
                if (i < j)
                {
                    tx = real[i];
                    real[i] = real[j];
                    real[j] = tx;

                    tx = complex[i];
                    complex[i] = complex[j];
                    complex[j] = tx;
                }

                k = i2;
                while (k <= j)
                {
                    j -= k;
                    k >>= 1;
                }
                j += k;
            }

            /* Compute the FFT */
            c1 = -1f;
            c2 = 0f;
            l2 = 1;
            for (l = 0; l < m; l++)
            {
                l1 = l2;
                l2 <<= 1;
                u1 = 1f;
                u2 = 0f;
                for (j = 0; j < l1; j++)
                {
                    for (i = j; i < nn; i += l2)
                    {
                        i1 = i + l1;
                        t1 = u1 * real[i1] - u2 * complex[i1];
                        t2 = u1 * complex[i1] + u2 * real[i1];
                        real[i1] = real[i] - t1;
                        complex[i1] = complex[i] - t2;
                        real[i] += t1;
                        complex[i] += t2;
                    }
                    z = u1 * c1 - u2 * c2;
                    u2 = u1 * c2 + u2 * c1;
                    u1 = z;
                }
                c2 = MathF.Sqrt((1f - c1) / 2f);
                if (dir == 1) c2 = -c2;
                c1 = MathF.Sqrt((1f + c1) / 2f);
            }

            /* Scaling for forward transform */
            if (dir == 1)
            {
                for (i = 0; i < nn; i++)
                {
                    real[i] /= nn;
                    complex[i] /= nn;
                }
            }

            return true;
        }


        /// <summary>
        /// Calculate the phase of the signal
        /// </summary>
        /// <param name="real">the real part</param>
        /// <param name="complex">the complex part</param>
        /// <returns>the phase</returns>
        public static float[] CalcPhase(float[] real, float[] complex)
        {
            if (real == null || complex == null) return null;
            if (complex.Length != real.Length) return null;

            float[] phase = new float[real.Length];
            for (int i = 0; i < real.Length; i++)
            {
                phase[i] = MathF.Atan2(complex[i], real[i]);
            }

            return phase;
        }

        /// <summary>
        /// Calculate the amplitude of the signal 
        /// </summary>
        /// <param name="real">the real part</param>
        /// <param name="complex">the complex part</param>
        /// <returns>the amplitude</returns>
        public static float[] CalcAmplitude(float[] real, float[] complex)
        {
            return MathEx.Distance(real, complex);
        }



    }
}
