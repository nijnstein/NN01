using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using NSS;
using NSS.Imaging;

namespace NN01
{
    public static class Threshold
    {
        /// <summary>
        /// Locally Adaptive Thresholder  (white backgrounds only!!)
        /// </summary>
        /// <param name="bmp_in">the input bitmap of minimal neighbourhood size, must be either, 8,24 or 32 bits per pixel</param>
        /// <param name="neighbourhood_size">the size of the neighbourhood to use for selecting a thresshold.. hmm TODO could allow rectangular sizes, they should give better results for vertical gradients.. from scanning with a plate and a bad doc holder</param>
        /// <returns>the thresholded 1 bit bitmap</returns>
        public static Bitmap AdaptiveThreshold(this Bitmap bmp_in, uint neighbourhood_size)
        {
            if (bmp_in == null) return null;
            if (bmp_in.PixelFormat != PixelFormat.Format24bppRgb) return null;
            if (bmp_in.Width < neighbourhood_size || bmp_in.Height < neighbourhood_size) return null;
            PixelFormat[] supported_formats = new PixelFormat[4]
            {
    PixelFormat.Format8bppIndexed,
    PixelFormat.Format24bppRgb,
    PixelFormat.Format32bppArgb,
    PixelFormat.Format32bppRgb
            };
            if (!supported_formats.Contains(bmp_in.PixelFormat)) return null;

            Bitmap bmp_out = new Bitmap(bmp_in.Width, bmp_in.Height, PixelFormat.Format1bppIndexed);
            Rectangle rcLock = new Rectangle(0, 0, bmp_in.Width, bmp_out.Width);

            BitmapData bmd_in = bmp_in.LockBits(rcLock, ImageLockMode.ReadOnly, bmp_in.PixelFormat);
            try
            {
                BitmapData bmd_out = bmp_out.LockBits(rcLock, ImageLockMode.ReadWrite, PixelFormat.Format1bppIndexed);
                try
                {
                    unsafe
                    {
                        void* in_scan0 = bmd_in.Scan0.ToPointer();
                        void* out_scan0 = bmd_out.Scan0.ToPointer();

                        LocalAdaptiveThreshold
                        (
                         in_scan0, bmd_in.Stride,
                         out_scan0, bmd_out.Stride,
                         (uint)bmp_in.Width,
                         (uint)bmp_in.Height,
                         neighbourhood_size,
                         (uint)bmp_in.PixelFormat.ToBitDepth()
                        );

                    }
                }
                finally
                {
                    bmp_out.UnlockBits(bmd_out);
                }
            }
            finally
            {
                bmp_in.UnlockBits(bmd_in);
            }

            return bmp_out;
        }

        private static int Neighbourhood_AvgMin = 0;
        private static int Neighbourhood_AvgMax = 150;

        private static unsafe void PrecalcRowMinMax(
         byte* p_row, int row_width, int segment_width, int* min, int* max, uint sample_depth)
        {
            if (min == null) return;
            if (max == null) return;

            int ii = 0;
            int imin = 255 * 3;
            int imax = 0;

            int avg_min = Neighbourhood_AvgMin;
            int avg_max = Neighbourhood_AvgMax;

            if (sample_depth > 8)
            {
                avg_max = avg_max * 3;
                avg_min = avg_min * 3;
            }

            for (int x = 0; x < row_width; x++)
            {
                int t = 0;
                if (sample_depth >= 24)
                {
                    byte b = *p_row++;
                    byte g = *p_row++;
                    byte r = *p_row++;
                    if (sample_depth == 32) p_row++;
                    t = r + g + b;
                }
                else
                 if (sample_depth == 8)
                {
                    t = *p_row++;
                }

                if (t < imin) imin = t;
                if (t > imax) imax = t;

                ii++;
                if (ii == segment_width)
                {
                    // min and max may be the same when there is a massive solid object
                    // orr.. nothing just whitespace..
                    // assume whitespace to be higher in value!! TODO -> need to test this with 8-bit grayscale may fail when indexed..
                    if (imin.InMargin(imax, 2, 2))
                    {
                        if (imin < avg_min + 10)
                        {
                            // inside a solid object dark, use averaged values
                            imin = avg_min;
                            imax = avg_max;
                        }
                        else
                         if (imax >= avg_max && imin > avg_max >> 1)
                        {
                            // white
                            imin = avg_min;
                            imax = avg_max;
                        }
                    }

                    *min++ = imin;
                    *max++ = imax;
                    ii = 0;
                    avg_min = imin + avg_min >> 1;
                    avg_max = imax + avg_max >> 1;
                    imin = 255 * 3;
                    imax = 0;
                }
            }
        }


        /// <summary>
        /// Local Adaptive Threshold Algorithm
        /// - precalculated neighbourhood :O
        /// </summary>
        /// <param name="in_scan0"></param>
        /// <param name="in_stride"></param>
        /// <param name="out_scan0"></param>
        /// <param name="out_stride"></param>
        /// <param name="neighbourhood_size"></param>
        public static unsafe bool LocalAdaptiveThreshold(
         void* in_scan0, int in_stride, void* out_scan0, int out_stride,
         uint width, uint height, uint neighbourhood_size, uint in_sample_depth)
        {
            if (in_scan0 == null || out_scan0 == null) return false;
            if (width < neighbourhood_size || height < neighbourhood_size) return false;
            if (Math.Abs(in_stride) < width * 3) return false;
            if (Math.Abs(out_stride) < width / 32) return false;
            if (!(in_sample_depth == 8 || in_sample_depth == 24 || in_sample_depth == 32)) return false;

            // we buffer the output to the 1bpp bitmap in blocks of 32 bits/pixels 
            uint output_pixel = 0;

            // setup scanline pointers
            byte* in_scan = (byte*)in_scan0;
            uint* out_scan = (uint*)out_scan0;

            // the local threshold calculated and the neighbourhood
            int threshold = 0;

            uint nh2 = neighbourhood_size * neighbourhood_size;
            int ns2 = (int)(neighbourhood_size / 2);

            int segment_size = ns2;
            int segment_count = (int)((width + segment_size - 1) / segment_size);
            int* pnh_min = stackalloc int[(int)(segment_count * neighbourhood_size)];
            int* pnh_max = stackalloc int[(int)(segment_count * neighbourhood_size)];
            int* seg_min = stackalloc int[segment_count];
            int* seg_max = stackalloc int[segment_count];

            // precalc min / max for segments on the x-axis  
            // - we dont need to do it _once_ for each pixel, instead of many times per pixel !!
            // - cost some memory, 
            //   roughly: width / (neigbourhood_size / 2) * neighbourhood_size * 4 bytes
            for (int i = 0; i < neighbourhood_size; i++)
            {
                byte* p_row = in_scan + i * in_stride;
                int* pmin = pnh_min + i * segment_count;
                int* pmax = pnh_max + i * segment_count;

                PrecalcRowMinMax(p_row, (int)width, segment_size, pmin, pmax, in_sample_depth);
            }
            int nh_current_update = 0;

            // threshold the imagedata!!
            for (int y = 0; y < height; y++)
            {
                int i_segment = 0;
                int ii_segment = 0;
                bool b_invalidate_threshold = true;
                bool b_invalidate_segments = true;

                int min = 255 * 3;
                int max = 0;

                byte* p_in = in_scan;
                uint* p_out = out_scan;

                int yn = (int)Math.Max(0, Math.Min(y - ns2, height - neighbourhood_size - 1));

                // need to update neighbourhood? 
                if (y > ns2 && y < height - ns2)
                {
                    // just cycle and replace the oldest row: 
                    // - the y doesnt matter as we are always at the center!!
                    // - the order of processing also does not matter for the neighbourhood as were using minmax
                    int* pmin = pnh_min + nh_current_update * segment_count;
                    int* pmax = pnh_max + nh_current_update * segment_count;
                    byte* prow = in_scan - ns2 * in_stride;

                    PrecalcRowMinMax(prow, (int)width, segment_size, pmin, pmax, in_sample_depth);

                    b_invalidate_segments = true;

                    nh_current_update++;
                    if (nh_current_update == neighbourhood_size) nh_current_update = 0;
                }

                if (b_invalidate_segments)
                {
                    // precalc column totals
                    int* pmin = seg_min;
                    int* pmax = seg_max;
                    for (int i = 0; i < segment_count; i++)
                    {
                        *pmin++ = 255 * 3;
                        *pmax++ = 0;
                    }

                    for (int j = 0; j < neighbourhood_size; j++)
                    {
                        pmin = seg_min;
                        pmax = seg_max;
                        int* pmax2 = pnh_max + segment_count * j;
                        int* pmin2 = pnh_min + segment_count * j;
                        for (int i = 0; i < segment_count; i++)
                        {
                            int imin = *pmin;
                            int imax = *pmax;
                            int imin2 = *pmin2;
                            int imax2 = *pmax2;
                            if (imin2 < imin) *pmin = imin2;
                            if (imax2 > imax) *pmax = imax2;
                            pmin++;
                            pmin2++;
                            pmax++;
                            pmax2++;
                        }
                    }
                }

                // thresshold the pixels in the row.. 
                for (int x = 0; x < width; x++)
                {
                    int bgr = 0;

                    // get current pixel 
                    if (in_sample_depth == 24)
                    {
                        byte b = *p_in++;
                        byte g = *p_in++;
                        byte r = *p_in++;
                        bgr = b + g + r;
                    }
                    else
                     if (in_sample_depth == 32)
                    {
                        byte b = *p_in++;
                        byte g = *p_in++;
                        byte r = *p_in++;
                        p_in++;
                        bgr = b + g + r;
                    }
                    else
                      if (in_sample_depth == 8)
                    {
                        bgr = *p_in++;
                    }

                    // calc minmax from neighbourhood.. 
                    if (b_invalidate_threshold)
                    {
                        min = 255 * 3;
                        max = 0;
                        int* pmin = seg_min + i_segment;
                        int* pmax = seg_max + i_segment;

                        // first segment
                        if (i_segment > 0)
                        {
                            if (*--pmin < min) min = *pmin;
                            pmin++;
                            if (*--pmax > max) max = *pmax;
                            pmax++;
                        }

                        // current segment
                        if (*pmin < min) min = *pmin;
                        if (*pmax > max) max = *pmax;

                        // the last
                        if (i_segment < segment_count - 1)
                        {
                            if (*++pmin < min) min = *pmin;
                            if (*++pmax > max) max = *pmax;
                        }

                        threshold = (max + min >> 1) - 12; // = 3 * 4 :)
                        b_invalidate_threshold = false;
                    }

                    // update segment index if needed
                    ii_segment++;
                    if (ii_segment == segment_size)
                    {
                        i_segment++;
                        ii_segment = 0;
                        b_invalidate_threshold = true;
                    }

                    // compare current pixel 
                    int current_pixel = x & 31;
                    if (bgr < threshold)
                    {
                        // write black pixel
                        output_pixel &= ~(uint)(1 << 31 - current_pixel);
                    }
                    else
                    {
                        // write white pixel 
                        output_pixel |= (uint)(1 << 31 - current_pixel);
                    }

                    // write output pixel
                    if (current_pixel == 31 || width < 31 && x == width - 1)
                    {
                        // shuffle pixels
                        output_pixel = (output_pixel & 0x000000FF) << 24 |
                                       (output_pixel & 0x0000FF00) << 8 |
                                       (output_pixel & 0x00FF0000) >> 8 |
                                       (output_pixel & 0xFF000000) >> 24;

                        *p_out++ = output_pixel;

                        output_pixel = 0;
                    }
                }


                in_scan += in_stride;
                out_scan += out_stride >> 2;
            }

            return true;
        }

    }
}
