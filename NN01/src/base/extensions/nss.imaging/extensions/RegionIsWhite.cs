using System.Drawing;
using System.Drawing.Imaging;

namespace NSS.Imaging
{
    public static class BitmapIsWhite
    {

        static public bool IsWhiteRegion(this Bitmap bmp, Rectangle rc)
        {
            return RegionIsWhite(bmp, rc);
        }

        static public bool RegionIsWhite(Bitmap bmp, Rectangle rc)
        {
            BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, bmp.PixelFormat);
            bool b = RegionIsWhite(bmp, bmd, rc);
            bmp.UnlockBits(bmd);
            return b;
        }

        static public bool RegionIsWhite(Bitmap bmp, BitmapData bmd, Rectangle rc)
        {
            unsafe
            {
                byte* pScan = (byte*)bmd.Scan0;
                int iThreshold = 240;

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
                                    if ((*p & mask) != wmask) return false;
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
                                        return false;
                                    }
                                    ++p;

                                    for (uint x = x1 + 1; x < x2 - 1; x++)
                                    {
                                        if (*p != white_value)
                                        {
                                            return false;
                                        }
                                        ++p;
                                    }

                                    if ((*p & end_mask) != (white_value & end_mask))
                                    {
                                        return false;
                                    }

                                    pScan += bmd.Stride;
                                }
                            }
                            return true;
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
                                        if (*p != white_value) return false;
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
                                            // TODO  this is _very_ expensive.. should init a boolean lookup  iswhite[256] initialized from whitelist
                                            // if(iswhite + *p != 0) return false; 
                                            if (white_list.Contains((byte)*p)) return false;
                                        }
                                        ++p;
                                    }
                                }
                            }
                            else
                              if (white_list.Count == 0)
                            {
                                // no white color in palette
                                return false;
                            }
                            white_list = null;
                            return true;
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
                                    if (*(p + 0) < iThreshold) return false;
                                    if (*(p + 1) < iThreshold) return false;
                                    if (*(p + 2) < iThreshold) return false;

                                    //skip alpha
                                    p += 4;  // TODO: not sure, may need to be first depending on order of bits
                                }
                                pScan += bmd.Stride;
                            }
                            return true;
                        }

                    case PixelFormat.Format24bppRgb:
                        {
                            pScan += bmd.Stride * rc.Top + rc.Left * 3;
                            for (int y = rc.Top; y < rc.Bottom; y++)
                            {
                                byte* p = pScan;
                                for (int x = rc.Left; x < rc.Bottom; x++)
                                {
                                    if (*(p + 0) < iThreshold) return false;
                                    if (*(p + 1) < iThreshold) return false;
                                    if (*(p + 2) < iThreshold) return false;
                                    p += 3;
                                }
                                pScan += bmd.Stride;
                            }
                            return true;
                        }
                }
            }
            throw new NotSupportedException("Unsupported Pixel Format");
        }

    }
}
