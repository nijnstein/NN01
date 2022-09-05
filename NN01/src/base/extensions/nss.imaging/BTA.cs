using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;

namespace NSS.Imaging
{
    public enum BTA_Colormode : int
    {
        /// <summary>
        /// == R+G+B / 3    == GrayScale
        /// </summary>
        Lightness,
        /// <summary>
        /// == 0.2126 R + 0.7152 G + 0.0722 B
        /// </summary>
        Luminance,
        /// <summary>
        ///  R,G,B // OUTPUT is 3 times larger!!
        /// </summary>
        Color,
        /// <summary>
        /// Red Channel Only
        /// </summary>
        RedChannel,
        /// <summary>
        /// Green Channel Only
        /// </summary>
        GreenChannel,
        /// <summary>
        /// Blue Channel Only
        /// </summary>
        BlueChannel
    }

    public static class BTA
    {
        #region btda scanners 
        /// <summary>
        /// Convert the pixel data into a double array, white = 0.0  black = 1.0
        /// </summary>
        /// <param name="scan0"></param>
        /// <param name="stride"></param>
        /// <param name="output"></param>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        public static unsafe void BitmapToDoubleArray_1BPP(uint* scan0, int stride, double* output, int x1, int y1, int x2, int y2)
        {
            // calc the mask needed to mask out bits at the start and end of the scanline
            int x1_32 = x1 + 31 >> 5;
            uint x1_mask = ~((uint)(1 << Math.Abs(x1 - (x1 + 31 >> 5) << 5)) - 1);

            int x2_32 = x2 + 31 >> 5;
            uint x2_mask = (uint)(1 << Math.Abs(x2 - (x2 + 31 >> 5) << 5)) - 1;

            // for each row in the pixel data
            for (int y = y1; y < y2; y++)
            {
                uint* pSampleScan = scan0 + stride / 4 * y1 + x1_32;

                // scan pixelrow, in blocks of 32 pixels. 
                for (int x = x1_32; x < x2_32; x++)
                {
                    uint s32 = ~*pSampleScan;

                    if (s32 == 0)
                    {
                        // add 32 zeros to the output..
                        for (int i = 0; i < 32; i++)
                        {
                            *output = 0.0;
                            output++;
                        }
                    }
                    else
                    {
                        // swap bytes in a workable order
                        s32 = (s32 & 0x000000FF) << 24 |
                              (s32 & 0x0000FF00) << 8 |
                              (s32 & 0x00FF0000) >> 8 |
                              (s32 & 0xFF000000) >> 24;

                        int low = 0;
                        int high = 31;

                        // mask out last bytes on start and end of scanline.. not a nice place for this..
                        // TODO split this outside the loop
                        if (x == x1_32)
                        {
                            s32 = s32 & x1_mask;
                            // bound the bit search below.. 
                        }
                        else
                         if (x == x2_32 - 1)
                        {
                            s32 = s32 & x2_mask;
                        }

                        // scan for pixels in the sample
                        for (int bit = high; bit >= low; bit--)
                        {
                            if ((s32 & (uint)(1 << bit)) > 0)
                            {
                                // have pixel! add a 1
                                *output = 1.0;
                            }
                            else
                            {
                                // no pixel
                                *output = 0.0;
                            }

                            output++;
                        }
                    }

                    pSampleScan++;
                }
            }
        }

        /// <summary>
        /// Convert the pixel data into a double array using the specified color mode 
        /// </summary>
        /// <param name="scan0"></param>
        /// <param name="stride"></param>
        /// <param name="output"></param>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        public static unsafe void BitmapToDoubleArray_24BPP(byte* scan0, int stride, double* output, int x1, int y1, int x2, int y2, BTA_Colormode color_mode)
        {
            byte* pScan = scan0 + y1 * stride + x1 * 3;
            for (int y = y1; y < y2; y++)
            {
                byte* p = pScan;

                switch (color_mode)
                {
                    case BTA_Colormode.Lightness:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                byte b = *p;
                                p++;
                                byte g = *p;
                                p++;
                                byte r = *p;
                                p++;
                                *output++ = (b + g + r) / 3.0;
                            }
                            break;
                        }

                    case BTA_Colormode.Luminance:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                byte b = *p;
                                p++;
                                byte g = *p;
                                p++;
                                byte r = *p;
                                p++;
                                *output++ = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                            }
                            break;
                        }

                    case BTA_Colormode.Color:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                byte b = *p++;
                                *output++ = b;
                                byte g = *p++;
                                *output++ = g;
                                byte r = *p++;
                                *output++ = r;
                                *output++ = (r + g + b) / 3.0;
                            }
                            break;
                        }
                    case BTA_Colormode.RedChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                int m = *p++ + *p++ >> 1;
                                byte r = *p++;
                                *output++ = MathEx.Clip255(r - m);
                            }
                            break;
                        }
                    case BTA_Colormode.GreenChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                int m = *p++;
                                byte g = *p++;
                                m += *p++;
                                m = m >> 1;
                                *output++ = MathEx.Clip255(g - m);
                            }
                            break;
                        }
                    case BTA_Colormode.BlueChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                byte b = *p++;
                                int m = *p++ + *p++ >> 1;
                                *output++ = MathEx.Clip255(b - m);
                            }
                            break;
                        }
                }
                pScan += stride;
            }
        }

        public static unsafe void BitmapToDoubleArray_32BPP(uint* scan0, int stride, double* output, int x1, int y1, int x2, int y2, BTA_Colormode color_mode)
        {
            uint* pScan = scan0 + y1 * (stride / 4) + x1;

            for (int y = y1; y < y2; y++)
            {
                uint* p = pScan;
                switch (color_mode)
                {
                    case BTA_Colormode.Lightness:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = (c->R + c->G + c->B) / 3.0;
                            }
                            break;
                        }
                    case BTA_Colormode.Luminance:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = 0.2126 * c->R + 0.7152 * c->G + 0.0722 * c->B;
                            }
                            break;
                        }
                    case BTA_Colormode.Color:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = ~c->R;
                                *output++ = ~c->G;
                                *output++ = ~c->B;
                                *output++ = (~c->R + ~c->G + ~c->B) / 3.0;
                            }
                            break;
                        }
                    case BTA_Colormode.RedChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = c->R;
                            }
                            break;
                        }
                    case BTA_Colormode.GreenChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = c->R;
                            }
                            break;
                        }
                    case BTA_Colormode.BlueChannel:
                        {
                            for (int x = x1; x < x2; x++)
                            {
                                Color32* c = (Color32*)p++;
                                *output++ = c->R;
                            }
                            break;
                        }
                }
                pScan += stride / 4;
            }
        }

        public static unsafe void BitmapToDoubleArray(PixelFormat pixel_format, void* scan0, int stride, double* pout, int x1, int y1, int x2, int y2, BTA_Colormode color_mode)
        {
            switch (pixel_format)
            {
                case PixelFormat.Format1bppIndexed:
                    {
                        BitmapToDoubleArray_1BPP((uint*)scan0, stride, pout, x1, y1, x2, y2);
                        break;
                    }

                case PixelFormat.Format24bppRgb:
                    {
                        BitmapToDoubleArray_24BPP((byte*)scan0, stride, pout, x1, y1, x2, y2, color_mode);
                        break;
                    }

                case PixelFormat.Format32bppRgb:
                case PixelFormat.Format32bppArgb:
                    {
                        BitmapToDoubleArray_32BPP((uint*)scan0, stride, pout, x1, y1, x2, y2, color_mode);
                        break;
                    }

                default:
                    throw new NotImplementedException();
            }
        }

        #endregion

        /// <summary>
        /// Crop out a portion of bitmap and convert it into a double array of hue values,
        /// ranging 0..1 / 0.0 .. 255.09//??????? TODO
        /// </summary>
        /// <param name="bmp"></param>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        /// <returns></returns>
        public static double[] BitmapToDoubleArray(Bitmap bmp, int x1, int y1, int x2, int y2, BTA_Colormode color_mode)
        {
            if (x1 > x2) MathEx.Swap(ref x1, ref x2);
            if (y1 > y2) MathEx.Swap(ref y1, ref y2);

            int w = x2 - x1;
            int h = y2 - y1;
            int n = w * h;
            if (color_mode == BTA_Colormode.Color) n *= 4;

            BitmapData bmd = bmp.LockBits(new Rectangle(x1, y1, w, h), ImageLockMode.ReadOnly, bmp.PixelFormat);
            try
            {
                double[] output = new double[n];

                unsafe
                {
                    void* scan0 = bmd.Scan0.ToPointer();
                    fixed (double* pout = output)
                    {
                        BitmapToDoubleArray(bmp.PixelFormat, scan0, bmd.Stride, pout, x1, y1, x2, y2, color_mode);
                    }
                }

                return output;
            }
            finally
            {
                bmp.UnlockBits(bmd);
            }
        }

        /// <summary>
        /// generates a 24bpp grayscale bitmap from the doubles
        /// </summary>
        /// <param name="da"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static Bitmap FromDoubleArray(double[] da, int width, int height, bool shift, BTA_Colormode color_mode)
        {
            if (da == null) return null;

            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    fixed (double* pd0 = da)
                    {
                        byte* scan0 = (byte*)bmd.Scan0.ToPointer();
                        double* pd = pd0;

                        // if shift then shift the corners to the center :)
                        if (shift)
                        {
                            for (int y = 0; y < height; y++)
                            {
                                int y2 = (y + height / 2) % height;
                                switch (color_mode)
                                {
                                    case BTA_Colormode.Lightness:
                                    case BTA_Colormode.Luminance:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                double d = *pd++;
                                                byte c = (byte)MathEx.Clip255((int)d);

                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;

                                                *p++ = c;
                                                *p++ = c;
                                                *p++ = c;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.Color:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                double d = *pd++;
                                                byte b = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte g = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte r = (byte)MathEx.Clip255((int)d);
                                                pd++;

                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;

                                                *p++ = b;
                                                *p++ = g;
                                                *p++ = r;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.RedChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = 0;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.BlueChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.GreenChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                }
                            }
                        }
                        else
                        {
                            for (int y = 0; y < height; y++)
                            {
                                byte* p = scan0;

                                switch (color_mode)
                                {
                                    case BTA_Colormode.Lightness:
                                    case BTA_Colormode.Luminance:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                double d = *pd++;
                                                byte c = (byte)MathEx.Clip255((int)d);

                                                *p++ = c;
                                                *p++ = c;
                                                *p++ = c;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.Color:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                double d = *pd++;
                                                byte b = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte g = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte r = (byte)MathEx.Clip255((int)d);
                                                pd++;

                                                *p++ = b;
                                                *p++ = g;
                                                *p++ = r;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.RedChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = 0;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.BlueChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.GreenChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                }
                                scan0 += bmd.Stride;
                            }
                        }
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bmd);
            }

            return bmp;
        }

        public static Bitmap FromFloatArray(float[] a, int width, int height, bool shift, BTA_Colormode color_mode)
        {
            if (a == null) return null;

            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    fixed (float* pd0 = a)
                    {
                        byte* scan0 = (byte*)bmd.Scan0.ToPointer();
                        float* pd = pd0;

                        // if shift then shift the corners to the center :)
                        if (shift)
                        {
                            for (int y = 0; y < height; y++)
                            {
                                int y2 = (y + height / 2) % height;
                                switch (color_mode)
                                {
                                    case BTA_Colormode.Lightness:
                                    case BTA_Colormode.Luminance:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                float d = *pd++;
                                                byte c = (byte)MathEx.Clip255((int)d);

                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;

                                                *p++ = c;
                                                *p++ = c;
                                                *p++ = c;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.Color:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                float d = *pd++;
                                                byte b = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte g = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte r = (byte)MathEx.Clip255((int)d);
                                                pd++;

                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;

                                                *p++ = b;
                                                *p++ = g;
                                                *p++ = r;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.RedChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = 0;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.BlueChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.GreenChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                int x2 = (x + width / 2) % width;
                                                byte* p = scan0 + y2 * bmd.Stride + x2 * 3;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                }
                            }
                        }
                        else
                        {
                            for (int y = 0; y < height; y++)
                            {
                                byte* p = scan0;

                                switch (color_mode)
                                {
                                    case BTA_Colormode.Lightness:
                                    case BTA_Colormode.Luminance:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                float d = *pd++;
                                                byte c = (byte)MathEx.Clip255((int)d);

                                                *p++ = c;
                                                *p++ = c;
                                                *p++ = c;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.Color:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                float d = *pd++;
                                                byte b = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte g = (byte)MathEx.Clip255((int)d);
                                                d = *pd++;
                                                byte r = (byte)MathEx.Clip255((int)d);
                                                pd++;

                                                *p++ = b;
                                                *p++ = g;
                                                *p++ = r;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.RedChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = 0;
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.BlueChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                    case BTA_Colormode.GreenChannel:
                                        {
                                            for (int x = 0; x < width; x++)
                                            {
                                                *p++ = 0;
                                                *p++ = (byte)MathEx.Clip255((int)*pd++);
                                                *p++ = 0;
                                            }
                                            break;
                                        }
                                }
                                scan0 += bmd.Stride;
                            }
                        }
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bmd);
            }

            return bmp;
        }


        public static Bitmap ToBitmap(this double[] data, int w, int h)
        {
            return FromDoubleArray(data, w, h, false, BTA_Colormode.Lightness);
        }

        public static Bitmap ToBitmap(this double[] data, int w, int h, int low, int high)
        {
            return data.ToBitmap(w, h, low, high, false, false);
        }

        public static Bitmap ToBitmap(this double[] data, int w, int h, int low, int high, bool inplace)
        {
            return data.ToBitmap(w, h, low, high, false, inplace);
        }

        public static Bitmap ToBitmap(this double[] data, int w, int h, int low, int high, bool logarithmic, bool inplace)
        {
            if (!inplace) data = (double[])data.Clone();
            if (logarithmic)
                data.NormalizeLog10(low, high);
            else
                data.Normalize(low, high);
            return FromDoubleArray(data, w, h, false, BTA_Colormode.Lightness);
        }



        public static Bitmap ToBitmap(this float[] data, int w, int h)
        {
            return FromFloatArray(data, w, h, false, BTA_Colormode.Lightness);
        }

        public static Bitmap ToBitmap(this float[] data, int w, int h, int low, int high)
        {
            return data.ToBitmap(w, h, low, high, false, false);
        }

        public static Bitmap ToBitmap(this float[] data, int w, int h, int low, int high, bool inplace)
        {
            return data.ToBitmap(w, h, low, high, false, inplace);
        }

        public static Bitmap ToBitmap(this float[] data, int w, int h, int low, int high, bool logarithmic, bool inplace)
        {
            if (!inplace) data = (float[])data.Clone();
            if (logarithmic)
                data.NormalizeLog10(low, high);
            else
                data.Normalize(low, high);
            return FromFloatArray(data, w, h, false, BTA_Colormode.Lightness);
        }


    }
}
