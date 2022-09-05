using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS.Imaging 
{
    public class ConvolutionFilter
    {
        /*  3x3 Convolution Masks

      laplace      hipass     find edges   sharpen    edge enhance  color emboss
                              (top down)                            (well, kinda)
      -1 -1 -1    -1 -1 -1     1  1  1     -1 -1 -1     0 -1  0       1  0  1
      -1  8 -1    -1  9 -1     1 -2  1     -1 16 -1    -1  5 -1       0  0  0
      -1 -1 -1    -1 -1 -1    -1 -1 -1     -1 -1 -1     0 -1  0       1  0 -2

          1           1           1            8           1             1

       Soften        blur    Soften (less)

       2  2  2     3  3  3     0  1  0
       2  0  2     3  8  3     1  2  1
       2  2  2     3  3  3     0  1  0

         16          32           6
      }         
                */
        static public int[] C3_LaPlace = { -1, -1, -1, -1, 8, -1, -1, -1, -1, 1 };
        static public int[] C3_HiPass = { -1, -1, -1, -1, 9, -1, -1, -1, -1, 1 };
        static public int[] C3_RemoveBackground = { 2, 2, 2, 2, 8, 2, 2, 2, 2, 16 };
        static public int[] C3_RemoveBackgroundLess = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 16 };
        static public int[] C3_Sharpen = { -1, -1, -1, -1, 16, -1, -1, -1, -1, 8 };

        static public Bitmap Convolve3(Bitmap bmp_in, Rectangle rcRegion, int[] mask)
        {
            BitmapData bmd = bmp_in.LockBits(new Rectangle(0, 0, bmp_in.Width, bmp_in.Height), ImageLockMode.ReadWrite, bmp_in.PixelFormat);
            try
            {
                Bitmap bmp_out = new Bitmap(bmp_in.Width, bmp_in.Height, bmp_in.PixelFormat);
                BitmapData bmd_out = bmp_out.LockBits(new Rectangle(0, 0, bmp_out.Width, bmp_out.Height), ImageLockMode.ReadWrite, bmp_out.PixelFormat);
                try
                {
                    return Convolve3(bmp_in, bmd, bmp_out, bmd_out, rcRegion, mask);
                }
                finally
                {
                    bmp_out.UnlockBits(bmd_out);
                }
            }
            finally
            {
                bmp_in.UnlockBits(bmd);
            }
        }

        // 3x3 Convolution Filter
        static public Bitmap Convolve3(Bitmap bmp_in, BitmapData bmd_in, Bitmap bmp_out, BitmapData bmd_out, Rectangle rcRegion, int[] mask)
        {
            if (bmp_in == null) throw new ArgumentNullException("bmp_in");
            if (bmd_in == null) throw new ArgumentNullException("bmd_in");
            if (bmp_out == null) throw new ArgumentNullException("bmp_out");
            if (bmd_out == null) throw new ArgumentNullException("bmd_out");
            if ((bmp_in.Width != bmp_out.Width) || (bmp_in.Height != bmp_out.Height))
                throw new ArgumentOutOfRangeException("bmp", "The input and output bitmap must be of the same size!");
            if (bmp_in.PixelFormat != bmp_out.PixelFormat)
                throw new ArgumentOutOfRangeException("bmp", "The input and output bitmap must have the same pixel format!");

            if (mask == null) throw new ArgumentNullException("mask");
            if (mask.Length < 10) throw new ArgumentOutOfRangeException("mask", "Should be at least 3x3 + divider, with x,y = 2n-1");

            // allow masks of sizes 3,5,7...
            int size = (int)Math.Sqrt(mask.Length);
            if ((size <= 1) || (((size - 1) & 1) == 1)) throw new ArgumentOutOfRangeException("mask", "Should be at least 3x3 + divider, with x,y = 2n-1");

            int x1 = rcRegion.Left + 1;
            int y1 = rcRegion.Top + 1;
            int x2 = rcRegion.Right - 1;
            int y2 = rcRegion.Bottom - 1;

            switch (bmd_in.PixelFormat)
            {
                case PixelFormat.Format1bppIndexed:
                    {
                        Convolve3_1BPP(bmp_in, bmd_in, bmp_out, bmd_out, x1, y1, x2, y2, mask);
                        break;
                    }

                case PixelFormat.Format24bppRgb:
                    {
                        Convolve3_24BPP(bmd_in, bmd_out, x1, y1, x2, y2, mask);
                        break;
                    }

                case PixelFormat.Format32bppRgb:
                    {
                        Convolve3_32RGB(bmd_in, bmd_out, x1, y1, x2, y2, mask);
                        break;
                    }

                default: throw new Exception("Convolve: Pixelformat not supported!");
            }
            return bmp_out;
        }

        /// <summary>
        /// 1BPP 3x3 Convolution Filter  
        /// </summary>
        /// <param name="bmp"></param>
        /// <param name="bmd"></param>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        /// <param name="mask"></param>
        static private void Convolve3_1BPP(Bitmap bmp_in, BitmapData bmd_in, Bitmap bmp_out, BitmapData bmd_out, int x1, int y1, int x2, int y2, int[] mask)
        {
            unsafe
            {
                byte* pScan = (byte*)bmd_in.Scan0.ToPointer() + bmd_in.Stride * y1 + (x1 >> 3);
                uint white_value = IndexedColor.GetWhiteValue_1BPP(bmp_in);
                int stridediv4 = bmd_in.Stride >> 2;

                byte* pScanOut = (byte*)bmd_out.Scan0.ToPointer() + bmd_out.Stride * y1 + (x1 >> 3);

                int m0 = mask[0];
                int m1 = mask[1];
                int m2 = mask[2];
                int m3 = mask[3];
                int m4 = mask[4];
                int m5 = mask[5];
                int m6 = mask[6];
                int m7 = mask[7];
                int m8 = mask[8];
                int divider = mask[9];

                int mr0 = 0;
                int mr1 = 0;
                int mr2 = 0;
                int mr3 = 0;
                int mr4 = 0;
                int mr5 = 0;
                int mr6 = 0;
                int mr7 = 0;
                int mr8 = 0;

                for (int y = y1; y <= y2; y++)
                {
                    uint* p = (uint*)pScan;
                    uint* p_prev = null;
                    uint* p_out = (uint*)pScanOut;
                    uint* p_out_prev = null;

                    uint u = 0, u_out_prev = 0, u_out = 0;
                    byte b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0, b8 = 0;

                    for (int x = x1; x < x2;)
                    {
                        u_out_prev = u_out;
                        u_out = 0;

                        uint* p_sprev = p - stridediv4;
                        uint* p_snext = p + stridediv4;

                        uint u_sprev;
                        uint u_snext;

                        if (white_value > 0)
                        {
                            u = ~(*p);
                            u_sprev = ~(*p_sprev);
                            u_snext = ~(*p_snext);
                        }
                        else
                        {
                            u = *p;
                            u_sprev = *p_sprev;
                            u_snext = *p_snext;
                        }

                        int i = x & 31;
                        for (; (i <= 31) && (x < x2); i++)
                        {
                            uint m = (uint)(1 << (31 - i));
                            // [ 0  1  2 ] 
                            // [ 3  4  5 ]     4 = current pixel
                            // [ 6  7  8 ]  
                            b0 = b1;
                            b1 = b2;
                            b2 = (byte)((u_sprev & m) > 0 ? 1 : 0);
                            b3 = b4;
                            b4 = b5;
                            b5 = (byte)((u & m) > 0 ? 1 : 0);
                            b6 = b7;
                            b7 = b8;
                            b8 = (byte)((u_snext & m) > 0 ? 1 : 0);

                            mr0 = (m0 * b0);
                            mr1 = (m1 * b1);
                            mr2 = (m2 * b2);
                            mr3 = (m3 * b3);
                            mr4 = (m4 * b4);
                            mr5 = (m5 * b5);
                            mr6 = (m6 * b6);
                            mr7 = (m7 * b7);
                            mr8 = (m8 * b8);

                            float f = (mr0 + mr1 + mr2 + mr3 + mr4 + mr5 + mr6 + mr7 + mr8) / (float)divider;

                            // remember, take samples at b5! from p*..
                            // may need to go 1 back!
                            if (f > 0.9)
                            {
                                if (i > 0)
                                {
                                    u_out |= (uint)1 << (32 - i);
                                }
                                else
                                {
                                    // write on last bit of previous pixel 
                                    u_out_prev |= (uint)1;
                                }
                            }
                            else
                            {
                                // set black value
                                if (i > 0)
                                {
                                    u_out &= ~(uint)(1 << (32 - i));
                                }
                                else
                                {
                                    // write on last byte of previous pixel 
                                    u_out_prev &= ~(uint)((uint)1);
                                }
                            }

                            x++; // next pixel
                        } // /for i in pixel block

                        // write back pixel block
                        if (white_value > 0)
                        {
                            *p_out = ~u_out;
                            if (p_out_prev != null) *p_out_prev = (~u_out_prev);
                        }
                        else
                        {
                            *p_out = u_out;
                            if (p_out_prev != null) *p_out_prev = u_out_prev;
                        }
                        p_out_prev = p_out;
                        p_prev = p;

                        // next 32 pixel block
                        p++;
                        p_out++;
                    }

                    pScan += bmd_in.Stride;
                    pScanOut += bmd_out.Stride;
                }
            }
        }




        static private void Convolve3_24BPP(BitmapData bmd_in, BitmapData bmd_out, int x1, int y1, int x2, int y2, int[] mask)
        {
            unsafe
            {
                byte* pScan = (byte*)bmd_in.Scan0.ToPointer() + (y1 * bmd_in.Stride) + (x1 * 3);

                // 1 sample behind!
                byte* pOut = (byte*)bmd_out.Scan0.ToPointer() + (y1 * bmd_out.Stride) + (x1 - 1) * 3;

                float fdivider = mask[9];

                Color24 back;
                back.R = 0;
                back.G = 0;
                back.B = 0;

                int m0 = mask[0];
                int m1 = mask[1];
                int m2 = mask[2];
                int m3 = mask[3];
                int m4 = mask[4];
                int m5 = mask[5];
                int m6 = mask[6];
                int m7 = mask[7];
                int m8 = mask[8];
                int divider = mask[9];

                for (int y = y1; y <= y2; y++)
                {
                    byte* p = pScan;
                    byte* p_out = pOut;
                    byte* p_prev = pScan - bmd_in.Stride;
                    byte* p_next = pScan + bmd_in.Stride;

                    Color24 b, b0, b1, b2, b3, b4, b5, b6, b7, b8;
                    b = back;
                    b0 = back;
                    b1 = back;
                    b2 = back;
                    b3 = back;
                    b4 = back;
                    b5 = back;
                    b6 = back;
                    b7 = back;
                    b8 = back;

                    for (int x = x1 - 1; x <= x2 - 1; x++)
                    {
                        b0 = b1;
                        b1 = b2;
                        b3 = b4;
                        b4 = b5;
                        b6 = b7;
                        b7 = b8;

                        b2.R = p_prev[3];
                        b2.G = p_prev[4];
                        b2.B = p_prev[5];
                        b5.R = p[3];
                        b5.G = p[4];
                        b5.B = p[5];
                        b8.R = p_next[3];
                        b8.G = p_next[4];
                        b8.B = p_next[5];

                        b.R = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.R) +
                                (m1 * b1.R) +
                                (m2 * b2.R) +
                                (m3 * b3.R) +
                                (m4 * b4.R) +
                                (m5 * b5.R) +
                                (m6 * b6.R) +
                                (m7 * b7.R) +
                                (m8 * b8.R)) / divider);      //  todo ? implement as shifter -> only dividers of 2^n 

                        b.G = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.G) +
                                (m1 * b1.G) +
                                (m2 * b2.G) +
                                (m3 * b3.G) +
                                (m4 * b4.G) +
                                (m5 * b5.G) +
                                (m6 * b6.G) +
                                (m7 * b7.G) +
                                (m8 * b8.G)) / divider);

                        b.B = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.B) +
                                (m1 * b1.B) +
                                (m2 * b2.B) +
                                (m3 * b3.B) +
                                (m4 * b4.B) +
                                (m5 * b5.B) +
                                (m6 * b6.B) +
                                (m7 * b7.B) +
                                (m8 * b8.B)) / divider);

                        p_prev += 3;
                        p += 3;
                        p_next += 3;

                        *p_out = b.R;
                        p_out++;
                        *p_out = b.G;
                        p_out++;
                        *p_out = b.B;
                        p_out++;
                    }

                    pScan += bmd_in.Stride;
                    pOut += bmd_out.Stride;
                }
            }
        }



        static private void Convolve3_32RGB(BitmapData bmd_in, BitmapData bmd_out, int x1, int y1, int x2, int y2, int[] mask)
        {
            unsafe
            {
                byte* pScan = (byte*)bmd_in.Scan0.ToPointer() + (y1 * bmd_in.Stride) + (x1 * 4);

                // 1 sample behind!
                byte* pOut = (byte*)bmd_out.Scan0.ToPointer() + (y1 * bmd_out.Stride) + (x1 - 1) * 4;

                float fdivider = mask[9];

                Color32 back;
                back.ARGB = 0;
                back.R = 0;
                back.G = 0;
                back.B = 0;
                back.A = 255;

                int m0 = mask[0];
                int m1 = mask[1];
                int m2 = mask[2];
                int m3 = mask[3];
                int m4 = mask[4];
                int m5 = mask[5];
                int m6 = mask[6];
                int m7 = mask[7];
                int m8 = mask[8];
                int divider = mask[9];

                for (int y = y1; y <= y2; y++)
                {
                    byte* p = pScan;
                    byte* p_out = pOut;
                    byte* p_prev = pScan - bmd_in.Stride;
                    byte* p_next = pScan + bmd_in.Stride;

                    Color32 b, b0, b1, b2, b3, b4, b5, b6, b7, b8;
                    b = back;
                    b0 = back;
                    b1 = back;
                    b2 = back;
                    b3 = back;
                    b4 = back;
                    b5 = back;
                    b6 = back;
                    b7 = back;
                    b8 = back;

                    for (int x = x1 - 1; x <= x2 - 1; x++)
                    {
                        b0 = b1;
                        b1 = b2;
                        b3 = b4;
                        b4 = b5;
                        b6 = b7;
                        b7 = b8;

                        b2.R = p_prev[3];
                        b2.G = p_prev[4];
                        b2.B = p_prev[5];
                        b5.R = p[3];
                        b5.G = p[4];
                        b5.B = p[5];
                        b8.R = p_next[3];
                        b8.G = p_next[4];
                        b8.B = p_next[5];

                        b.R = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.R) +
                                (m1 * b1.R) +
                                (m2 * b2.R) +
                                (m3 * b3.R) +
                                (m4 * b4.R) +
                                (m5 * b5.R) +
                                (m6 * b6.R) +
                                (m7 * b7.R) +
                                (m8 * b8.R)) / divider);      //  todo ? implement as shifter -> only dividers of 2^n 

                        b.G = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.G) +
                                (m1 * b1.G) +
                                (m2 * b2.G) +
                                (m3 * b3.G) +
                                (m4 * b4.G) +
                                (m5 * b5.G) +
                                (m6 * b6.G) +
                                (m7 * b7.G) +
                                (m8 * b8.G)) / divider);

                        b.B = (byte)
                               MathEx.Clip255(
                               ((m0 * b0.B) +
                                (m1 * b1.B) +
                                (m2 * b2.B) +
                                (m3 * b3.B) +
                                (m4 * b4.B) +
                                (m5 * b5.B) +
                                (m6 * b6.B) +
                                (m7 * b7.B) +
                                (m8 * b8.B)) / divider);

                        p_prev += 4;
                        p += 4;
                        p_next += 4;

                        *p_out = 255; // alpha component
                        p_out++;
                        *p_out = b.R;
                        p_out++;
                        *p_out = b.G;
                        p_out++;
                        *p_out = b.B;
                        p_out++;
                    }

                    pScan += bmd_in.Stride;
                    pOut += bmd_out.Stride;
                }
            }
        }

    }
}
