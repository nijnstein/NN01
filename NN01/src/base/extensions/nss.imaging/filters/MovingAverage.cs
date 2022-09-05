
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NSS.Imaging
{

    public class MovingAverage : KernelFilter
    {
        public override bool Apply(Bitmap bmp, int kernel_size)
        {
            if (bmp == null) return false;

            switch (bmp.PixelFormat)
            {
                case PixelFormat.Format24bppRgb:
                    {
                        return MovingAverage_24BPP(bmp, kernel_size);
                    }
            }

            return false;
        }

        static public bool MovingAverage_24BPP(Bitmap bmp, int kernel_size)
        {
            if (bmp == null) return false;
            if (bmp.PixelFormat != PixelFormat.Format24bppRgb) return false;

            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width - 1, bmp.Height - 1), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {                    
                    byte* scan0 = (byte*)bmd.Scan0.ToPointer();
                    return MovingAverage_24BPP(scan0, bmd.Stride, bmp.Width, bmp.Height, kernel_size);
                }
            }
            finally
            {
                bmp.UnlockBits(bmd);
            }
        }

        static public unsafe bool MovingAverage_24BPP(byte* scan0, int stride, int width, int height, int kernel_size)
        {
            if (scan0 == null) return false;

            for (int y = 0; y < height; y++)
            {
                byte* pscan = scan0;

                for (int x = 0; x < width; x++)
                {
                    // calculate kernel 
                    int x1 = Math.Max(0, Math.Min(width - 1 - kernel_size, x - kernel_size / 2));
                    int x2 = x1 + kernel_size;

                    int b_kernel = 0;
                    int g_kernel = 0;
                    int r_kernel = 0;

                    /// todo  COULD MAKE THIS RECURSIVE
                    byte* pk = scan0 + x1 * 3;
                    for (int k = x1; k < x2; k++)
                    {
                        b_kernel += *pk++;
                        g_kernel += *pk++;
                        r_kernel += *pk++;
                    }

                    int s = x2 - x1 + 1;
                    b_kernel = (b_kernel + *pscan++) / s;
                    g_kernel = (g_kernel + *pscan++) / s;
                    r_kernel = (r_kernel + *pscan++) / s;

                    pscan -= 3;

                    // write back pixel
                    *pscan++ = MathEx.Clip255(b_kernel);
                    *pscan++ = MathEx.Clip255(g_kernel);
                    *pscan++ = MathEx.Clip255(r_kernel);
                }
                scan0 += stride;
            }

            return true;
        }

        public override int GetDefaultKernelSize()
        {
            return 5;
        }

        public override bool Supports(PixelFormat format)
        {
            return format == PixelFormat.Format24bppRgb;
        }

    }
}
