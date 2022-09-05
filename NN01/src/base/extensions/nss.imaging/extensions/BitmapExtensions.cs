using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;

namespace NSS.Imaging
{
    public enum ImageFilter
    {
        MovingAverage,
        SaltNPepper
    }

    public static class BitmapExtensions
    {
        #region Filter 
        static public bool ApplyFilter(this Bitmap bmp, ImageFilter filter)
        {
            switch (filter)
            {
                case ImageFilter.MovingAverage:
                    return (new MovingAverage()).Apply(bmp);

                case ImageFilter.SaltNPepper:
                    return (new SaltNPepper()).Apply(bmp);

                default: return false;
            }
        }

        static public Bitmap Convolve(this Bitmap bmp_in, Rectangle rcRegion, int[] mask)
        {
            return ConvolutionFilter.Convolve3(bmp_in, rcRegion, mask);
        }
        #endregion

        public static Bitmap Resize(this Bitmap _in, int w, int h)
        {
            return Resize(_in, w, h, true);
        }
        public static Bitmap Resize(this Bitmap _in, int w, int h, bool high_quality)
        {
            Bitmap _out = new Bitmap(w, h);
            using (Graphics x = Graphics.FromImage(_out))
            {
                if (high_quality)
                {
                    x.SmoothingMode = SmoothingMode.HighQuality;
                    x.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    x.CompositingQuality = CompositingQuality.HighQuality;
                }
                x.DrawImage(_in, 0, 0, w, h);
            }
            return _out;
        }


        static public int ToBitDepth(this System.Drawing.Imaging.PixelFormat format)
        {
            switch (format)
            {
                case PixelFormat.Format1bppIndexed: return 1;
                case PixelFormat.Format24bppRgb: return 24;
                case PixelFormat.Format32bppArgb: return 32;
                case PixelFormat.Format32bppPArgb: return 32;
                case PixelFormat.Format32bppRgb: return 32;
                case PixelFormat.Format16bppArgb1555: return 16;
                case PixelFormat.Format16bppGrayScale: return 16;
                case PixelFormat.Format16bppRgb555: return 16;
                case PixelFormat.Format16bppRgb565: return 16;
                case PixelFormat.Format8bppIndexed: return 8;
                case PixelFormat.Format48bppRgb: return 48;
                case PixelFormat.Format4bppIndexed: return 48;
                case PixelFormat.Format64bppArgb: return 64;
                case PixelFormat.Format64bppPArgb: return 64;
            }
            return 0;
        }

        #region Preparation 
        static public double[] ToDoubleArray(this Bitmap bmp, int x1, int y1, int x2, int y2, BTA_Colormode color_mode)
        {
            return BTA.BitmapToDoubleArray(bmp, x1, y1, x2, y2, color_mode);
        }
        #endregion

        #region ContentBorder
        static public RECT GetContentBorder(Bitmap bmp)
        {
            if (bmp != null)
            {
                BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width - 1, bmp.Height - 1), ImageLockMode.ReadOnly, bmp.PixelFormat);
                if (bmd != null)
                {
                    try
                    {
                        return GetContentBorder(bmp, bmd);
                    }
                    finally
                    {
                        bmp.UnlockBits(bmd);
                    }
                }
            }
            return RECT.Invalid;
        }

        static public RECT GetContentBorder(Bitmap bmp, BitmapData bmd)
        {
            switch (bmd.PixelFormat)
            {
                case PixelFormat.Format32bppArgb:
                    {
                        return GetContentBorder_32ARGB(bmp, bmd);
                    }
                case PixelFormat.Format24bppRgb:
                    {
                        return GetContentBorder_24RGB(bmp, bmd);
                    }
                case PixelFormat.Format8bppIndexed:
                    {
                        return GetContentBorder_8I(bmp, bmd);
                    }
                case PixelFormat.Format1bppIndexed:
                    {
                        return GetContentBorder_1BPP(bmp, bmd);
                    }
            }

            return RECT.Invalid;
        }

        static public RECT GetContentBorder_32ARGB(Bitmap bmp, BitmapData bmd)
        {
            RECT rcContent = new RECT(0, -1, 0, -1);
            unsafe
            {
                // setup scan pointers
                uint* p0 = (uint*)bmd.Scan0.ToPointer();
                uint* pScan = p0;

                // scan vertical borders, preserve cache line by doing the LTR and RTL scan on the same time
                int y = 0;
                while (y < bmd.Height)
                {
                    uint* p = pScan;

                    // left 
                    int x = 0;
                    int x2 = rcContent.Left == 0 ? bmd.Width : rcContent.Left;
                    while (x < x2)
                    {
                        uint s = *p & 0x00FFFFFF; // ignore alpha channel 
                        if (s > 0)
                        {
                            x2 = x;
                            rcContent.Left = x;
                        }
                        // else if left away, should be slightly faster with OoO execution
                        x++;
                        p++;
                    }

                    // right, skip if empty line
                    p = pScan + bmp.Width - 1;
                    if (x != bmd.Width)
                    {
                        x = rcContent.Right == 0 ? bmd.Width - 1 : rcContent.Right;
                        while (x >= rcContent.Left)
                        {
                            uint s = *p;
                            uint s2 = ((s & 0x00FF0000) >> 16) + ((s & 0x0000FF00) >> 8) + (s & 0x000000FF);
                            if (s < 750)
                            {
                                rcContent.Right = x;
                                x = rcContent.Left;
                            }
                            x--;
                            p--;
                        }
                    }

                    // set top and bottom borders
                    if ((rcContent.Left > 0) && (x == rcContent.Left))
                    {
                        if (rcContent.Top == -1)
                        {
                            rcContent.Top = y;
                        }
                        else
                        {
                            rcContent.Bottom = y;
                        }
                    }

                    y++;
                    pScan += (bmd.Stride / 4);
                }


            }
            return rcContent;
        }

        static public RECT GetContentBorder_24RGB(Bitmap bmp, BitmapData bmd)
        {
            RECT rcContent = new RECT(0, -1, 0, -1);
            unsafe
            {
                // setup scan pointers
                byte* p0 = (byte*)bmd.Scan0.ToPointer();
                byte* pScan = p0;

                // scan vertical borders, preserve cache line by doing the LTR and RTL scan on the same time
                int y = 0;
                while (y < bmd.Height)
                {
                    byte* p = pScan;

                    // left 
                    int x = 0;
                    int x2 = rcContent.Left == 0 ? bmd.Width : rcContent.Left;
                    while (x < x2)
                    {
                        uint s = *p;
                        p++;
                        s += *p;
                        p++;
                        s += *p;
                        p++;

                        if (s < 750) // approx white, todo 
                        {
                            x2 = x;
                            rcContent.Left = x;
                        }

                        x++;
                    }

                    // right, skip if empty line
                    if (x != bmd.Width)
                    {
                        p = pScan + (bmp.Width - 1) * 3 + 2;
                        x = bmp.Width - 1;
                        x2 = rcContent.Right == 0 ? rcContent.Left : rcContent.Right;
                        while (x >= x2)
                        {
                            uint s = *p;
                            p--;
                            s += *p;
                            p--;
                            s += *p;
                            p--;

                            if (s < 750)
                            {
                                rcContent.Right = x;
                                x = x2;
                            }
                            x--;
                        }
                    }

                    // set top and bottom borders
                    if ((rcContent.Left > 0) && (x == x2 - 1))
                    {
                        if (rcContent.Top == -1)
                        {
                            rcContent.Top = y;
                        }
                        else
                        {
                            rcContent.Bottom = y;
                        }
                    }

                    y++;
                    pScan += bmd.Stride;
                }
            }
            return rcContent;
        }

        static public RECT GetContentBorder_8I(Bitmap bmp, BitmapData bmd)
        {
            throw new NotImplementedException();
        }


        static public RECT GetContentBorder_1BPP(Bitmap bmp, BitmapData bmd)//, bool min_white_count)
        {
            return GetContentBorder_1BPP(bmp, bmd, 3);
        }

        static public RECT GetContentBorder_1BPP(Bitmap bmp, BitmapData bmd, int min_pixel_count)
        {
            if (bmp == null) throw new ArgumentNullException("bmp");
            if (bmd == null) throw new ArgumentNullException("bmd");
            if (bmd.PixelFormat != PixelFormat.Format1bppIndexed) throw new ArgumentOutOfRangeException("bmd.PixelFormat", "should be 1 bit per pixel");

            RECT rcContent = new RECT(int.MaxValue, -1, int.MinValue, -1);
            int ctop = 0;
            int cbottom = 0;

            unsafe
            {
                uint* p0 = (uint*)bmd.Scan0.ToPointer();
                int w32 = ((bmd.Width + 31) >> 5) - 1;

                // mask out last block
                uint mask = (uint)(1 << Math.Abs(bmd.Width - ((bmd.Width + 31) >> 5) << 5)) - 1;

                // scan vertical borders
                int y = 0;
                uint* pScan = p0;
                while (y < bmd.Height)
                {
                    uint* p = pScan;
                    bool bhit = false;

                    // LTR Scan (we need the max x to left where the first block of n blank/white pixels starts            
                    int x = 0;
                    bool b = true;
                    int c = 0;
                    while (x < w32 && b && x * 32 < rcContent.Left)
                    {
                        uint s32 = ~(*p);

                        switch (s32)
                        {
                            case 0:
                                break;
                            case 0xFFFFFFFF:
                                c += 32;
                                if (c >= min_pixel_count)
                                {
                                    // found left side of border 
                                    rcContent.Left = Math.Min(rcContent.Left, x * 32);
                                    b = false;
                                }
                                break;
                            default:
                                // block has pixel, swap bytes in a workable order.. 
                                s32 = ((s32 & 0x000000FF) << 24) |
                                      ((s32 & 0x0000FF00) << 8) |
                                      ((s32 & 0x00FF0000) >> 8) |
                                      ((s32 & 0xFF000000) >> 24);

                                // scan for pixels in the sample
                                for (int bit = 31; bit >= 0; bit--)
                                {
                                    if ((s32 & (uint)(1 << bit)) == 0)
                                    {
                                        c = 0;
                                    }
                                    else
                                    {
                                        // got pixel 
                                        c++;
                                        if (c >= min_pixel_count)
                                        {
                                            // found left side of border 
                                            rcContent.Left = Math.Min((x * 32) + 31 - bit - c, rcContent.Left);
                                            b = false;
                                            break;
                                        }
                                    }
                                }
                                break;
                        }
                        if (b)
                        {
                            x++;
                            p++;
                        }
                    }
                    if (!b) bhit = true;

                    // RTL Scan, must have found something in ltr otherwise rtl scanning is useless
                    if (x < w32)
                    {
                        int xr = w32 - 1;
                        p = pScan + w32;
                        c = 0;
                        b = true;

                        while (xr > x && xr * 32 >= rcContent.Right && b) // only scan until x or right border
                        {
                            uint s32 = ~(*p);

                            switch (s32)
                            {
                                case 0:
                                    break;
                                case 0xFFFFFFFF:
                                    c += 32;
                                    if (c >= min_pixel_count)
                                    {
                                        // found right side of border 
                                        rcContent.Right = Math.Max(rcContent.Right, xr * 32);
                                        b = false;
                                    }
                                    break;
                                default:
                                    // block has pixel, swap bytes in a workable order.. 
                                    s32 = ((s32 & 0x000000FF) << 24) |
                                          ((s32 & 0x0000FF00) << 8) |
                                          ((s32 & 0x00FF0000) >> 8) |
                                          ((s32 & 0xFF000000) >> 24);

                                    // scan for pixels in the sample
                                    for (int bit = 31; bit >= 0; bit--)
                                    {
                                        if ((s32 & (uint)(1 << bit)) == 0)
                                        {
                                            c = 0;
                                        }
                                        else
                                        {
                                            // got pixel 
                                            c++;
                                            if (c >= min_pixel_count)
                                            {
                                                // found left side of border 
                                                rcContent.Right = Math.Max((xr * 32) + bit + c, rcContent.Right);
                                                b = false;
                                                break;
                                            }
                                        }
                                    }
                                    break;
                            }
                            if (b)
                            {
                                xr--;
                                p--;
                            }
                        }
                        if (!b) bhit = true;

                        if (bhit)
                        {
                            if (rcContent.Top == -1)
                            {
                                ctop++;
                                if (ctop > min_pixel_count)
                                {
                                    rcContent.Top = y - ctop;
                                }
                            }
                            cbottom++;
                        }
                        else
                        {
                            ctop = 0;
                            if (cbottom > min_pixel_count)
                            {
                                rcContent.Bottom = y;
                            }
                            cbottom = 0;
                        }
                    }

                    // nxt line 
                    pScan += bmd.Stride / 4;
                    y++;
                }
            }
            return rcContent;
        }


        #endregion

        #region intensity vector

        static public int[] IntensityVector(this Bitmap bmp)
        {
            if (bmp == null) return new int[0];
            int[] h = new int[256];

            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
            unsafe
            {
                try
                {
                    byte* pScan = (byte*)bmd.Scan0.ToPointer();
                    switch (bmp.PixelFormat)
                    {
                        case PixelFormat.Format8bppIndexed:
                            {
                                for (int y = 0; y < bmp.Height; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 0; x < bmp.Width; x++) h[*p++]++;
                                    p += bmd.Stride;
                                }
                                break;
                            }
                        case PixelFormat.Format24bppRgb:
                            {
                                for (int y = 0; y < bmp.Height; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 0; x < bmp.Width; x++) h[(int)((*p++ + *p++ + *p++) / 3)]++;
                                    p += bmd.Stride;
                                }
                                break;
                            }
                        case PixelFormat.Format32bppArgb:
                            {
                                for (int y = 0; y < bmp.Height; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 0; x < bmp.Width; x++)
                                    {
                                        h[(int)((*p++ + *p++ + *p++) / 3)]++;
                                        p++;
                                    }
                                    p += bmd.Stride;
                                }
                                break;
                            }
                        case PixelFormat.Format1bppIndexed:
                            {
                                throw new NotSupportedException();
                            }
                    }
                }
                finally
                {
                    bmp.UnlockBits(bmd);
                }
            }
            return h;
        }

        #endregion

        #region Gradient Map 

        /// <summary>
        /// Gradient Mapping -> denoise / deblur pre-stage (the more intenser the gradient, the less blurry the image is)
        /// we should be able to train a network to learn a 2d convolution kernel to denoise/blur the input image perfectly
        /// the gradient map is of use as a measure of sharpness
        /// Gxy = [f(x+1,y) - f(x-1,y), f(x,y+1) - f(x,y-1)]
        /// Gradient= Sqrt(gx^2+gy^2) or G=|gx|+|gy|
        /// Force= ArcTan(gy/gx)
        /// </summary>
        /// <param name="bmp">Input image</param>
        /// <param name="G">Gradient</param>
        /// <param name="F">Force</param>
        /// <returns></returns>
        static public void GradientMap(this Bitmap bmp, out double[] G, out double[] F)
        {
            if (bmp == null) throw new ArgumentNullException("bmp");

            F = new double[bmp.Width * bmp.Height];
            G = new double[bmp.Width * bmp.Height];

            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
            unsafe
            {
                try
                {
                    byte* pScan = (byte*)bmd.Scan0.ToPointer();
                    switch (bmp.PixelFormat)
                    {
                        case PixelFormat.Format8bppIndexed:
                            {
                                pScan += bmd.Stride + 1;
                                for (int y = 1; y < bmp.Height - 1; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 1; x < bmp.Width - 1; x++)
                                    {
                                        double gx = *(p + 1) - *(p - 1);
                                        double gy = *(p + bmd.Stride) - *(p - bmd.Stride);
                                        if (gx == 0) gx++;
                                        int idx = y * bmp.Width + x;
                                        G[idx] = Math.Abs(gx) + Math.Abs(gy);   //  Math.Sqrt(gx * gx + gy * gy);
                                        F[idx] = Math.Atan(gy / gx);
                                        p++;
                                    }
                                    pScan += bmd.Stride;
                                }
                                break;
                            }

                        case PixelFormat.Format24bppRgb:
                            {
                                pScan += bmd.Stride + 3;
                                for (int y = 1; y < bmp.Height - 1; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 1; x < bmp.Width - 1; x++)
                                    {
                                        double gx = ((*(p + 3) + *(p + 4) + *(p + 5)) / 3) - ((*(p - 1) + *(p - 2) + *(p - 1)) / 3);
                                        double gy = (*(p + bmd.Stride) + *(p + bmd.Stride + 1) + *(p + bmd.Stride + 2)) / 3 - (*(p - bmd.Stride) + *(p - bmd.Stride + 1) + *(p - bmd.Stride + 2)) / 3;
                                        if (gx == 0) gx++;
                                        int idx = y * bmp.Width + x;
                                        G[idx] = Math.Abs(gx) + Math.Abs(gy);   //  Math.Sqrt(gx * gx + gy * gy);
                                        F[idx] = Math.Atan(gy / gx);
                                        p += 3;
                                    }
                                    pScan += bmd.Stride;
                                }
                                break;
                            }

                        case PixelFormat.Format32bppArgb:
                            {
                                pScan += bmd.Stride + 4;
                                for (int y = 1; y < bmp.Height - 1; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = 1; x < bmp.Width - 1; x++)
                                    {
                                        double gx = ((*(p + 3) + *(p + 4) + *(p + 5)) / 3) - ((*(p - 1) + *(p - 2) + *(p - 3)) / 3);
                                        double gy = (*(p + bmd.Stride) + *(p + bmd.Stride + 1) + *(p + bmd.Stride + 2)) / 3 - (*(p - bmd.Stride) + *(p - bmd.Stride + 1) + *(p - bmd.Stride + 2)) / 3;
                                        if (gx == 0) gx++;
                                        int idx = y * bmp.Width + x;
                                        G[idx] = Math.Abs(gx) + Math.Abs(gy);   //  Math.Sqrt(gx * gx + gy * gy);
                                        F[idx] = Math.Atan(gy / gx);
                                        p += 4;
                                    }
                                    pScan += bmd.Stride;
                                }
                                break;
                            }
                        case PixelFormat.Format1bppIndexed:
                            {
                                throw new NotSupportedException();
                            }
                    }
                }
                finally
                {
                    bmp.UnlockBits(bmd);
                }
            }
        }

        #endregion

        #region Get SetPixelCount


        static public int GetSetPixelCount(this Bitmap bmp, Rectangle rc)
        {
            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);
            int c = GetSetPixelCount(bmp, bmd, rc);
            bmp.UnlockBits(bmd);
            return c;
        }

        static public int GetSetPixelCount(Bitmap bmp, BitmapData bmd, Rectangle rc)
        {
            int iThreshold = 240;
            int c = 0;
            unsafe
            {
                byte* pScan = (byte*)bmd.Scan0;

                switch (bmd.PixelFormat)
                {
                    case PixelFormat.Format1bppIndexed:
                        {
                            // highest intensity = white 
                            uint white_value = IndexedColor.GetWhiteValue_1BPP(bmp);

                            uint x1 = (uint)rc.Left >> 5;
                            uint x2 = (uint)rc.Right >> 5;

                            uint start_mask = (uint)(1 << (rc.Left & 31)) - 1;
                            uint end_mask = ~(uint)((1 << (rc.Right & 31)) - 1);

                            pScan = pScan + (rc.Top * bmd.Stride) + (rc.Left >> 5) * 4;

                            if (x1 == x2)
                            {
                                uint* p = (uint*)pScan;
                                uint mask = start_mask | end_mask;
                                uint wmask = white_value & mask;
                                for (int y = rc.Top; y < rc.Bottom; y++)
                                {
                                    if ((*p & mask) != wmask) c += MathEx.CountBits(*p & wmask);
                                    p += bmd.Stride / 4;
                                }
                            }
                            else
                            {
                                for (int y = rc.Top; y < rc.Bottom; y++)
                                {
                                    uint* p = (uint*)pScan;

                                    if ((*p & start_mask) != (white_value & start_mask))
                                    {
                                        c += MathEx.CountBits(*p & start_mask);
                                    }
                                    ++p;

                                    for (uint x = x1 + 1; x < x2 - 1; x++)
                                    {
                                        if (*p != white_value)
                                        {
                                            c += MathEx.CountBits(*p);
                                        }
                                        ++p;
                                    }

                                    if ((*p & end_mask) != (white_value & end_mask))
                                    {
                                        c += MathEx.CountBits(*p & end_mask);
                                    }

                                    pScan += bmd.Stride;
                                }
                            }
                            return c;
                        }

                    case PixelFormat.Format8bppIndexed:
                        {
                            // allow multiple indices to have a white value.. 
                            byte min_white = 0;
                            byte max_white = 0;
                            List<byte> white_list = IndexedColor.GetWhiteValue_8BPP(bmp, ref min_white, ref max_white, iThreshold);
                            pScan += bmd.Stride * rc.Top + rc.Left;

                            if (white_list.Count == 1)
                            {
                                byte white_value = (byte)white_list[0];
                                for (int y = rc.Top; y < rc.Bottom; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = rc.Left; x < rc.Right; x++)
                                    {
                                        if (*p != white_value) c++;
                                        ++p;
                                    }
                                }
                            }
                            else
                             if (white_list.Count > 0)
                            {
                                for (int y = rc.Top; y < rc.Bottom; y++)
                                {
                                    byte* p = pScan;
                                    for (int x = rc.Left; x < rc.Right; x++)
                                    {
                                        byte b = *p;
                                        if ((b >= min_white) && (b <= max_white))
                                        {
                                            // this is _very_ expensive.. should init a boolean lookup  iswhite[256] initialized from whitelist
                                            // if(iswhite + *p != 0) return false; 
                                            if (white_list.Contains((byte)*p)) c++;
                                        }
                                        ++p;
                                    }
                                }
                            }
                            else
                              if (white_list.Count == 0)
                            {
                                // no white color in palette
                                return (int)(rc.Width * rc.Height);
                            }
                            white_list = null;
                            return c;
                        }

                    case PixelFormat.Format32bppArgb:
                    case PixelFormat.Format32bppRgb:
                        {
                            pScan += bmd.Stride * rc.Top + rc.Left * 4;
                            for (int y = rc.Top; y < rc.Bottom; y++)
                            {
                                byte* p = pScan;
                                for (int x = rc.Left; x < rc.Right; x++)
                                {
                                    if (*(p + 0) < iThreshold) c++;
                                    else
                                     if (*(p + 1) < iThreshold) c++;
                                    else
                                      if (*(p + 2) < iThreshold) c++;

                                    //skip alpha
                                    p += 4;  // TODO: not sure, may need to be first depending on order of bits
                                }
                                pScan += bmd.Stride;
                            }
                            return c;
                        }

                    case PixelFormat.Format24bppRgb:
                        {
                            pScan += bmd.Stride * rc.Top + rc.Left * 3;
                            for (int y = rc.Top; y < rc.Bottom; y++)
                            {
                                byte* p = pScan;
                                for (int x = rc.Left; x < rc.Bottom; x++)
                                {
                                    if (*(p + 0) < iThreshold) c++;
                                    else
                                     if (*(p + 1) < iThreshold) c++;
                                    else
                                      if (*(p + 2) < iThreshold) c++;
                                    p += 3;
                                }
                                pScan += bmd.Stride;
                            }
                            return c;
                        }
                }
            }
            throw new NotSupportedException("GetSetPixelCount - Unsupported Pixel Format");
        }

        #endregion

        static public Bitmap Copy(this Bitmap src, Rectangle section)
        {
            if (src == null || section == null || section == Rectangle.Empty) return null;
            if (src.PixelFormat == PixelFormat.Format1bppIndexed)
#if WIN32
                return Copy1BPP(src, section); 
#else
                throw new NotSupportedException("1 bpp bitmaps are not supported in Copy");
#endif

            Bitmap bmp = new Bitmap(section.Width, section.Height, src.PixelFormat);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.DrawImage(src, 0, 0, section, GraphicsUnit.Pixel);
            }
            return bmp;
        }
#if WIN32

        /// <summary>
        /// Copy a bitmap into a 1bpp of the same dimensions very fast (compared to normal .net code)
        /// </summary>
        static System.Drawing.Bitmap Copy1BPP(System.Drawing.Bitmap bmp_in, Rectangle section)
  {
   int w = bmp_in.Width, h = bmp_in.Height;
   IntPtr h_bmp_in = bmp_in.GetHbitmap();

   Win32.BITMAPINFO bmi = new Win32.BITMAPINFO();
   bmi.biSize = 40;  // the size of the BITMAPHEADERINFO struct
   bmi.biWidth = section.Width;
   bmi.biHeight = section.Height;
   bmi.biPlanes = 1; 
   bmi.biBitCount = 1; 
   bmi.biCompression = Win32.BI_RGB;  
   bmi.biSizeImage = (uint)(((w + 7) & 0xFFFFFFF8) * h / 8);
   bmi.biXPelsPerMeter = (int)(bmp_in.HorizontalResolution * (100.0 / 2.54));
   bmi.biYPelsPerMeter = (int)(bmp_in.VerticalResolution * (100.0 / 2.54));  
  
   bmi.biClrUsed = 2;
   bmi.biClrImportant = 2;
   bmi.cols = new uint[256];  
   bmi.cols[0] = Win32.MAKERGB(0, 0, 0); 
   bmi.cols[1] = Win32.MAKERGB(255, 255, 255); 

   IntPtr bits0;
   IntPtr h_bmp_out = Win32.CreateDIBSection(IntPtr.Zero, ref bmi, Win32.DIB_RGB_COLORS, out bits0, IntPtr.Zero, 0);
   IntPtr sdc = Win32.GetDC(IntPtr.Zero);
   IntPtr hdc = Win32.CreateCompatibleDC(sdc); Win32.SelectObject(hdc, h_bmp_in);
   IntPtr hdc0 = Win32.CreateCompatibleDC(sdc); Win32.SelectObject(hdc0, h_bmp_out);
   Win32.BitBlt(hdc0, 0, 0, section.Width, section.Height, hdc, section.Left, section.Top, Win32.SRCCOPY);

   System.Drawing.Bitmap bmp_out = System.Drawing.Bitmap.FromHbitmap(h_bmp_out);

   Win32.DeleteDC(hdc);
   Win32.DeleteDC(hdc0);
   Win32.ReleaseDC(IntPtr.Zero, sdc);
   Win32.DeleteObject(h_bmp_in);
   Win32.DeleteObject(h_bmp_out);

   return bmp_out;
  }

#endif

        /// <summary>
        /// draw a horizontal line on a single bit bitmap (.net does not support this on 1bpp bitmaps)
        /// </summary>
        /// <param name="bmd"></param>
        /// <param name="y"></param>
        static public void HLine_1BPP(this BitmapData bmd, uint y)
        {
            unsafe
            {
                uint* p0 = (uint*)bmd.Scan0.ToPointer();
                p0 += (bmd.Stride / 4) * y;
                int w32 = ((bmd.Width + 31) >> 5) - 1;
                for (int x = 0; x < w32; x++) *(p0++) = 0x00000000;
            }
        }



#if WIN32
        static public void SaveToTIFFFile(this Bitmap[] bmp, string filename)
  {
   MemoryStream ms = SaveToTIFFStream(bmp);
   if (ms != null)
    using (FileStream fs = File.Open(filename, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None))
    {
     ms.Position = 0;
     ms.WriteTo(fs);
    }
  }

  static public MemoryStream SaveToTIFFStream(this Bitmap[] bmp)
  {
   if (bmp != null && bmp.Length > 0)
   {
    ImageCodecInfo codec = null;
    ImageCodecInfo[] info = ImageCodecInfo.GetImageEncoders();

    for (int i = 0; i < info.Length; i++)
     if (info[i].FormatDescription.Equals("TIFF"))
     {
      codec = info[i];
      break;               // 29 okt 2:30
     }
    if (codec == null) throw new Exception("Invalid codec selection, TIFF not supported");

    MemoryStream ms = new MemoryStream();
    switch (bmp.Length)
    {
     case 1:
      // single page
      EncoderParameters ps = new EncoderParameters(1);
      ps.Param[0] = new EncoderParameter(
       System.Drawing.Imaging.Encoder.Compression,
       bmp[0].PixelFormat == PixelFormat.Format1bppIndexed ? (long)EncoderValue.CompressionCCITT4 : (long)EncoderValue.CompressionLZW
      );
      bmp[0].Save(ms, codec, ps);
      break;

     default:
      // multi page
      EncoderParameters p = new EncoderParameters(2);
      Bitmap b = null;

      for (int i = 0; i < bmp.Length; i++)
      {
       if (bmp[i] == null) continue;

       p.Param[0] = new EncoderParameter(
        System.Drawing.Imaging.Encoder.Compression, 
        bmp[0].PixelFormat == PixelFormat.Format1bppIndexed ? (long)EncoderValue.CompressionCCITT4 : (long)EncoderValue.CompressionLZW);
       
       p.Param[1] = new EncoderParameter(
        System.Drawing.Imaging.Encoder.SaveFlag,
        b == null ? (long)EncoderValue.MultiFrame : (long)EncoderValue.FrameDimensionPage);

       if (b == null)
       {
        b = bmp[i];
        b.Save(ms, codec, p);
       }
       else
        b.SaveAdd(bmp[i], p);
      }
      if (b != null)
      {
       p.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.SaveFlag, (long)EncoderValue.Flush);
       b.SaveAdd(p);
      }
      break;
    }
    return ms;
   }
   return null;
  }

#endif

    }
}
