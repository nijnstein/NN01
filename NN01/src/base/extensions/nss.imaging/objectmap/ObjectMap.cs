using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;

namespace NSS.Imaging
{



    public class MapObjectList : List<MapObject>
    {

    }


    public class ObjectMap : IDisposable
    {
        private uint[] m_Map = null;
        private MapObjectList m_Objects = null;
        private int m_MapWidth = 0;
        private int m_MapHeight = 0;

        public int ObjectCount
        {
            get
            {
                return m_Objects.Count;
            }
        }

        public int Width { get { return m_MapWidth; } }
        public int Height { get { return m_MapHeight; } }
        public uint[] Map { get { return m_Map; } }

        protected ObjectMap()
        {
            m_Map = null;
            m_Objects = null;
        }

        public IEnumerable<MapObject> Objects { get { return m_Objects; } }

        #region Object Mapping Helpers

        protected bool Init()
        {
            if (m_Objects == null)
                m_Objects = new MapObjectList();
            else
                m_Objects.Clear();

            uint u = (uint)(m_MapWidth * m_MapHeight);
            if (m_Map == null)
            {
                m_Map = new uint[u];
            }
            else
            {
                if (m_Map.Length < u)
                {
                    m_Map = new uint[u];
                }
            }

            return true;
        }

        protected void ReplaceMap(uint dwFind, uint dwReplace)
        {
            MapObject pFind = GetObject(dwFind);
            if (pFind != null) ReplaceMap(pFind, dwReplace);
        }

        public static unsafe void ReplaceMap(uint* pMap, int map_width, MapObject pFind, uint replace_id, MapObject pReplace)
        {
            if (pMap == null) return;
            if (pFind == null) return;

            uint* pScan = pMap + (map_width * pFind.Region.Top) + pFind.Region.Left;

            for (int y = pFind.Region.Top; y <= pFind.Region.Bottom; y++)
            {
                uint* pmap = pScan;
                for (int x = pFind.Region.Left; x <= pFind.Region.Right; x++)
                {
                    if (*pmap == pFind.Id)
                    {
                        *pmap = replace_id;
                    }
                    pmap++;
                }
                pScan += map_width;
            }

            if (replace_id == 0)
            {
            }
            else
            {
                if (pReplace != null)
                {
                    pReplace.Density = pReplace.Density + pFind.Density;
                    pReplace.Region.Left = Math.Min(pReplace.Region.Left, pFind.Region.Left);
                    pReplace.Region.Right = Math.Max(pReplace.Region.Right, pFind.Region.Right);
                    pReplace.Region.Top = Math.Min(pReplace.Region.Top, pFind.Region.Top);
                    pReplace.Region.Bottom = Math.Max(pReplace.Region.Bottom, pFind.Region.Bottom);
                }
            }
        }

        protected void ReplaceMap(MapObject pFind, uint dwReplace)
        {
            if (pFind == null) return;

            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    MapObject pReplace = GetObject(dwReplace);

                    ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, pFind, dwReplace, pReplace);

                    pFind.Density = 0;
                    pFind.Id = 0;
                    m_Objects.Remove(pFind);
                }
            }
        }

        #region scanners
        protected void Scan1(BitmapData bmpData, RECT rcRegion, uint MaxObjects)
        {
            if (bmpData == null) throw new ArgumentNullException("bmpData");
            if (RECT.IsNullOrEmpty(rcRegion)) return;

            unsafe
            {
                // TODO for simplicity the rcRegion is ignored for know 
                fixed (uint* pFixedMap = m_Map)
                {
                    // fix the map and scan blocks of 32 bit
                    uint* pMapScan = (uint*)pFixedMap; // starting at (0,0) 
                    uint* pScan = (uint*)bmpData.Scan0.ToPointer();

                    // calc the mask needed to mask out bits at the end of the scanline
                    int w32 = (bmpData.Width + 31) >> 5;
                    uint mask = (uint)(1 << Math.Abs(bmpData.Width - ((bmpData.Width + 31) >> 5) << 5)) - 1;

                    // label from topleft to bottomright
                    for (int y = 0; y < bmpData.Height; y++)
                    {
                        uint* pmap = pMapScan;
                        uint* pSampleScan = pScan;

                        // scan pixelrow, in blocks of 32 pixels. 
                        for (int x = 0; x < w32; x++)
                        {
                            uint s32 = ~(*pSampleScan);

                            if (s32 != 0)
                            {
                                // swap bytes in a workable order.. could skip but im no machine it gets to messy.. todo
                                s32 = ((s32 & 0x000000FF) << 24) |
                                      ((s32 & 0x0000FF00) << 8) |
                                      ((s32 & 0x00FF0000) >> 8) |
                                      ((s32 & 0xFF000000) >> 24);

                                // mask out last bytes on end of scanline.. not a nice place for this..
                                if (x == (w32 - 1)) s32 = s32 & mask;

                                // scan for pixels in the sample
                                for (int bit = 31; bit >= 0; bit--)
                                {
                                    if ((s32 & (uint)(1 << bit)) > 0)
                                    {
                                        // have pixel! already an object around that position, note the map borders!
                                        uint uObject = 0;

                                        if (x > 0 || bit < 31)
                                        {
                                            uObject = pmap[-1];

                                            if (uObject == 0 && y > 0)
                                            {
                                                uObject = pmap[-m_MapWidth - 1];
                                                if (uObject == 0)
                                                {
                                                    uObject = pmap[-m_MapWidth];
                                                    if (uObject == 0)
                                                    {
                                                        if (bit > 0 || x < w32) uObject = pmap[-m_MapWidth + 1];
                                                    }
                                                }
                                            }
                                        }

                                        // new object? 
                                        if (uObject == 0)
                                        {
                                            uObject = (uint)m_Objects.Count + 1;
                                            if (uObject > MaxObjects) return; //quit

                                            MapObject pObj = new MapObject();

                                            UpdateObjectRegion(pObj, (x * 32) + 31 - bit, y);

                                            pObj.Id = uObject;
                                            pObj.Region.Left = (x * 32) + 31 - bit;
                                            pObj.Region.Right = pObj.Region.Left;
                                            pObj.Region.Top = y;
                                            pObj.Region.Bottom = y;
                                            pObj.Density = 1;
                                            // pObj.DensityCenterX = pObj.Region.Left * 2;
                                            // pObj.DensityCenterY = pObj.Region.Left * 2; 


                                            m_Objects.Add(pObj);
                                        }
                                        else
                                        {
                                            // existing object, we can index here because at this point: ID = idx+1
                                            MapObject pObj = m_Objects[(int)uObject - 1];

                                            int nx = (x * 32) + 31 - bit;

                                            pObj.Region.Left = Math.Min(nx, pObj.Region.Left);
                                            pObj.Region.Right = Math.Max(nx, pObj.Region.Right);
                                            pObj.Region.Top = Math.Min(y, pObj.Region.Top);
                                            pObj.Region.Bottom = Math.Max(y, pObj.Region.Bottom);


                                            pObj.Density = pObj.Density + 1;
                                        }
                                        *pmap = uObject;
                                    }
                                    ++pmap;
                                }
                            }
                            else
                            {
                                pmap += 32;
                            }

                            pSampleScan++;
                        }
                        // next map scanline
                        pMapScan += m_MapWidth;
                        pScan += bmpData.Stride / 4;
                    }
                }
            }
        }

        private void UpdateObjectRegion(MapObject pObj, int x, int y)
        {

        }

        /// <summary>
        /// 8-bit per sample indexed map scanner.. 
        /// note: the color palette is NOT used, 
        /// TODO: could extend to match on colors, that would cost a 
        ///       table lookup + 3 diffs for each pixel
        /// </summary>
        /// <param name="bmpData"></param>
        /// <param name="rc"></param>
        protected void Scan8(BitmapData bmpData, RECT rcRegion)
        {
            if (bmpData == null) throw new ArgumentNullException("bmpData");

            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    uint* pMapScan = (uint*)pFixedMap + (m_MapWidth * rcRegion.Top) + rcRegion.Left;
                    byte* pScan = (byte*)bmpData.Scan0.ToPointer() + (bmpData.Stride * rcRegion.Top) + (rcRegion.Left * 4);

                    // in the old day the top left pixel was used to indicate the background color.. 
                    // TODO  scan for the background color / probably white and the one of the most abundant (black may be 1)
                    byte white_index = *pScan;

                    // label from topleft to bottomright
                    for (int y = rcRegion.Top; y <= rcRegion.Bottom; y++)
                    {
                        uint* pmap = pMapScan;
                        byte* pSampleScan = pScan;

                        for (int x = rcRegion.Left; x <= rcRegion.Right; x++)
                        {
                            byte* pSample = (byte*)pSampleScan;
                            byte sample = *pSample;

                            // if the sample is non-white.. 
                            if (sample != white_index)
                            {
                                uint uObject = pmap[-m_MapWidth - 1];

                                if (uObject == 0)
                                {
                                    uObject = pmap[-m_MapWidth];
                                    if (uObject == 0)
                                    {
                                        uObject = pmap[-m_MapWidth + 1];
                                        if (uObject == 0)
                                        {
                                            uObject = pmap[-1];
                                        }
                                    }
                                }

                                if (uObject == 0)
                                {
                                    // new object
                                    uObject = (uint)m_Objects.Count + 1;

                                    MapObject pObj = new MapObject();

                                    pObj.Id = uObject;
                                    pObj.Region.Left = x;
                                    pObj.Region.Right = x;
                                    pObj.Region.Top = y;
                                    pObj.Region.Bottom = y;
                                    pObj.Density = 1;

                                    m_Objects.Add(pObj);
                                }
                                else
                                {
                                    // existing object, we can index here because at this point: ID = idx+1
                                    MapObject pObj = m_Objects[(int)uObject - 1];

                                    pObj.Region.Left = Math.Min(x, pObj.Region.Left);
                                    pObj.Region.Right = Math.Max(x, pObj.Region.Right);
                                    pObj.Region.Top = Math.Min(y, pObj.Region.Top);
                                    pObj.Region.Bottom = Math.Max(y, pObj.Region.Bottom);
                                    pObj.Density = pObj.Density + 1;
                                }
                                *pmap = uObject;
                            }
                            ++pmap;
                            pSampleScan += 1;
                        }
                        // next map scanline
                        pMapScan += m_MapWidth;
                        pScan += bmpData.Stride;
                    }
                }
            }
        }

        protected void Scan24(BitmapData bmpData, RECT rcRegion)
        {
            if (bmpData == null) throw new ArgumentNullException("bmpData");
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    uint* pMapScan = (uint*)pFixedMap + (m_MapWidth * rcRegion.Top) + rcRegion.Left;
                    byte* pScan = (byte*)bmpData.Scan0.ToPointer() + (bmpData.Stride * rcRegion.Top) + (rcRegion.Left * 3);

                    // label from topleft to bottomright
                    for (int y = rcRegion.Top; y <= rcRegion.Bottom; y++)
                    {
                        uint* pmap = pMapScan;
                        byte* pSampleScan = pScan;

                        for (int x = rcRegion.Left; x <= rcRegion.Right; x++)
                        {
                            Color24* pSample = (Color24*)pSampleScan;

                            // todo : this should be based on the background color intensity /hue  
                            if ((pSample->R + pSample->G + pSample->B) < (3 * 220))
                            {
                                uint uObject = pmap[-m_MapWidth - 1];

                                if (uObject == 0)
                                {
                                    uObject = pmap[-m_MapWidth];
                                    if (uObject == 0)
                                    {
                                        uObject = pmap[-m_MapWidth + 1];
                                        if (uObject == 0)
                                        {
                                            uObject = pmap[-1];
                                        }
                                    }
                                }

                                if (uObject == 0)
                                {
                                    // new object
                                    uObject = (uint)m_Objects.Count + 1;

                                    MapObject pObj = new MapObject();

                                    pObj.Id = uObject;
                                    pObj.Region.Left = x;
                                    pObj.Region.Right = x;
                                    pObj.Region.Top = y;
                                    pObj.Region.Bottom = y;
                                    pObj.Density = 1;

                                    m_Objects.Add(pObj);
                                }
                                else
                                {
                                    // existing object, we can index here because at this point: ID = idx+1
                                    MapObject pObj = m_Objects[(int)uObject - 1];

                                    pObj.Region.Left = Math.Min(x, pObj.Region.Left);
                                    pObj.Region.Right = Math.Max(x, pObj.Region.Right);
                                    pObj.Region.Top = Math.Min(y, pObj.Region.Top);
                                    pObj.Region.Bottom = Math.Max(y, pObj.Region.Bottom);
                                    pObj.Density = pObj.Density + 1;
                                }
                                *pmap = uObject;
                            }
                            ++pmap;
                            pSampleScan += 3;
                        }
                        // next map scanline
                        pMapScan += m_MapWidth;
                        pScan += bmpData.Stride;
                    }
                }
            }
        }

        protected unsafe void ScanSample(double[] input_sample, int input_width, int input_height, RECT rcRegion)
        {
            fixed (uint* pFixedMap = m_Map)
            {
                uint* pMapScan = (uint*)pFixedMap + (m_MapWidth * rcRegion.Top) + rcRegion.Left;

                // label from topleft to bottomright
                for (int y = rcRegion.Top; y <= rcRegion.Bottom; y++)
                {
                    uint* pmap = pMapScan;

                    for (int x = rcRegion.Left; x <= rcRegion.Right; x++)
                    {
                        bool bSample = input_sample[x + y * input_width] > 0;

                        if (bSample)
                        {
                            uint uObject = pmap[-m_MapWidth - 1];

                            if (uObject == 0)
                            {
                                uObject = pmap[-m_MapWidth];
                                if (uObject == 0)
                                {
                                    uObject = pmap[-m_MapWidth + 1];
                                    if (uObject == 0)
                                    {
                                        uObject = pmap[-1];
                                    }
                                }
                            }

                            if (uObject == 0)
                            {
                                // new object
                                uObject = (uint)m_Objects.Count + 1;

                                MapObject pObj = new MapObject();

                                pObj.Id = uObject;
                                pObj.Region.Left = x;
                                pObj.Region.Right = x;
                                pObj.Region.Top = y;
                                pObj.Region.Bottom = y;
                                pObj.Density = 1;

                                m_Objects.Add(pObj);
                            }
                            else
                            {
                                // existing object, we can index here because at this point: ID = idx+1
                                MapObject pObj = m_Objects[(int)uObject - 1];

                                pObj.Region.Left = Math.Min(x, pObj.Region.Left);
                                pObj.Region.Right = Math.Max(x, pObj.Region.Right);
                                pObj.Region.Top = Math.Min(y, pObj.Region.Top);
                                pObj.Region.Bottom = Math.Max(y, pObj.Region.Bottom);
                                pObj.Density = pObj.Density + 1;
                            }
                            *pmap = uObject;
                        }
                        ++pmap;
                    }
                    // next map scanline
                    pMapScan += m_MapWidth;
                }
            }
        }


        protected void Scan32(BitmapData bmpData, RECT rcRegion)
        {
            if (bmpData == null) throw new ArgumentNullException("bmpData");
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    uint* pMapScan = (uint*)pFixedMap + (m_MapWidth * rcRegion.Top) + rcRegion.Left;
                    byte* pScan = (byte*)bmpData.Scan0.ToPointer() + (bmpData.Stride * rcRegion.Top) + (rcRegion.Left * 4);

                    // label from topleft to bottomright
                    for (int y = rcRegion.Top; y <= rcRegion.Bottom; y++)
                    {
                        uint* pmap = pMapScan;
                        byte* pSampleScan = pScan;

                        for (int x = rcRegion.Left; x <= rcRegion.Right; x++)
                        {
                            Color32* pSample = (Color32*)pSampleScan;

                            // todo : this should be based on the background color intensity /hue  
                            if ((pSample->R + pSample->G + pSample->B) < (3 * 220))
                            {
                                uint uObject = pmap[-m_MapWidth - 1];

                                if (uObject == 0)
                                {
                                    uObject = pmap[-m_MapWidth];
                                    if (uObject == 0)
                                    {
                                        uObject = pmap[-m_MapWidth + 1];
                                        if (uObject == 0)
                                        {
                                            uObject = pmap[-1];
                                        }
                                    }
                                }

                                if (uObject == 0)
                                {
                                    // new object
                                    uObject = (uint)m_Objects.Count + 1;

                                    MapObject pObj = new MapObject();

                                    pObj.Id = uObject;
                                    pObj.Region.Left = x;
                                    pObj.Region.Right = x;
                                    pObj.Region.Top = y;
                                    pObj.Region.Bottom = y;
                                    pObj.Density = 1;

                                    m_Objects.Add(pObj);
                                }
                                else
                                {
                                    // existing object, we can index here because at this point: ID = idx+1
                                    MapObject pObj = m_Objects[(int)uObject - 1];

                                    pObj.Region.Left = Math.Min(x, pObj.Region.Left);
                                    pObj.Region.Right = Math.Max(x, pObj.Region.Right);
                                    pObj.Region.Top = Math.Min(y, pObj.Region.Top);
                                    pObj.Region.Bottom = Math.Max(y, pObj.Region.Bottom);
                                    pObj.Density = pObj.Density + 1;
                                }
                                *pmap = uObject;
                            }
                            ++pmap;
                            pSampleScan += 4;
                        }
                        // next map scanline
                        pMapScan += m_MapWidth;
                        pScan += bmpData.Stride;
                    }
                }
            }
        }

        #endregion

        public MapObject GetObject(uint ID)
        {
            // then search for it if not found
            foreach (MapObject p in m_Objects)
            {
                if (p == null) continue;
                if (p.Id == ID) return p;
            }
            return null;
        }

        protected unsafe void ReplaceObject(uint* pMap, uint find_id, uint replace_id)
        {
            MapObject pReplace = GetObject(replace_id);
            if (pReplace == null) return;

            MapObject pFind = GetObject(find_id);
            if (pFind == null) return;

            ReplaceObject(pMap, pFind, pReplace);
        }

        protected unsafe void ReplaceObject(uint* pMap, uint find_id, MapObject pReplace)
        {
            if (pMap == null) return;
            if (pReplace == null) return;

            MapObject pFind = GetObject(find_id);
            if (pFind == null) return;

            ReplaceObject(pMap, pFind, pReplace);
        }

        protected unsafe void ReplaceObject(uint* pMap, MapObject pFind, MapObject pReplace)
        {
            if (pMap == null) return;
            if (pReplace == null) return;
            if (pFind == null) return;

            ObjectMap.ReplaceMap(pMap, m_MapWidth, pFind, pReplace.Id, pReplace);

            pFind.Density = 0;

            if (m_Objects != null) m_Objects.Remove(pFind);
        }

        #endregion

        #region Map Generation | Create / Generate
        static public ObjectMap Create(Bitmap bmp, Rectangle rcRegion)
        {
            if (bmp == null) throw new ArgumentNullException("bmp");

            ObjectMap m = new ObjectMap();
            m.Generate(bmp, rcRegion);

            return m;
        }
        static public ObjectMap Create(double[] sample, int width, int height)
        {
            ObjectMap m = new ObjectMap();
            m.Generate(null, sample, width, height, Rectangle.FromLTRB(0, 0, width, height));

            return m;
        }
        static public ObjectMap Create(Bitmap bmp)
        {
            if (bmp == null) throw new ArgumentNullException("bmp");
            return ObjectMap.Create(bmp, new Rectangle(0, 0, bmp.Width - 1, bmp.Height - 1));
        }

        public bool Generate(Bitmap bmp, Rectangle rcRegion)
        {
            if (bmp == null) throw new ArgumentNullException("bmp");
            return Generate(bmp, null, bmp.Width, bmp.Height, rcRegion);
        }

        public bool Generate(Bitmap bmp, double[] input_sample, int input_width, int input_height, Rectangle rcRegion)
        {
            m_MapWidth = input_width;
            m_MapHeight = input_height;

            if (!Init()) return false;

            // discard surface borders.... 
            RECT rc = new RECT(
             Math.Max(1, rcRegion.Left),
             Math.Max(1, rcRegion.Top),
             Math.Min(input_width - 2, rcRegion.Right),
             Math.Min(input_height - 2, rcRegion.Bottom)
            );

            if (bmp != null)
            {
                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width - 1, bmp.Height - 1), ImageLockMode.ReadOnly, bmp.PixelFormat);
                try
                {
                    // label objects based on imagedata 
                    switch (bmp.PixelFormat)
                    {
                        case PixelFormat.Format1bppIndexed:
                            {
                                Scan1(bmpData, rc, uint.MaxValue);
                                break;
                            }

                        case PixelFormat.Format24bppRgb:
                            {
                                Scan24(bmpData, rc);
                                break;
                            }

                        case PixelFormat.Format8bppIndexed:
                            {
                                Scan8(bmpData, rc);
                                break;
                            }

                        case PixelFormat.Format32bppArgb:
                        case PixelFormat.Format32bppRgb:
                            {
                                Scan32(bmpData, rc);
                                break;
                            }

                        default: throw new Exception(string.Format("PixelFormat {0} is not supported", bmpData.PixelFormat.ToString()));
                    }
                }
                finally
                {
                    bmp.UnlockBits(bmpData);
                }
            }
            else
             if (input_sample != null)
            {
                ScanSample(input_sample, input_width, input_height, rc);
            }
            if (m_Objects.Count == 0) return false;

            // 2nd Pass: Connect Objects .. skipped


            // 3rd Pass: Now traverse map in the opposite direction of bitmap scan: bottom/right to top/left
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    for (int y = rc.Bottom; y > rc.Top; y--)
                    {
                        uint* pmap = pFixedMap + y * m_MapWidth + rc.Right;
                        for (int x = rc.Right; x > rc.Left; x--)
                        {
                            uint dw = *pmap;
                            if (dw != 0)
                            {
                                MapObject pReplace = null; // lazy load!!!!!! 

                                uint dw2 = pmap[-1];
                                if ((dw2 != dw) && (dw2 != 0))
                                {
                                    if (pReplace == null) pReplace = GetObject(dw);
                                    ReplaceObject(pFixedMap, dw2, pReplace);
                                }

                                dw2 = pmap[-1 - m_MapWidth];
                                if ((dw2 != dw) && (dw2 != 0))
                                {
                                    if (pReplace == null) pReplace = GetObject(dw);
                                    ReplaceObject(pFixedMap, dw2, pReplace);
                                }

                                dw2 = pmap[-m_MapWidth];
                                if ((dw2 != dw) && (dw2 != 0))
                                {
                                    if (pReplace == null) pReplace = GetObject(dw);
                                    ReplaceObject(pFixedMap, dw2, pReplace);
                                }

                                dw2 = pmap[-m_MapWidth + 1];
                                if ((dw2 != dw) && (dw2 != 0))
                                {
                                    if (pReplace == null) pReplace = GetObject(dw);
                                    ReplaceObject(pFixedMap, dw2, pReplace);
                                }

                                dw2 = pmap[m_MapWidth - 1];
                                if ((dw2 != dw) && (dw2 != 0))
                                {
                                    if (pReplace == null) pReplace = GetObject(dw);
                                    ReplaceObject(pFixedMap, dw2, pReplace);
                                }
                            }
                            --pmap;
                        }
                    }
                }
            }

            return m_Objects.Count > 0;
        }

        #endregion

        #region Map Object Content Border
        public Rectangle GetOuterContentBorder()
        {
            int x = m_MapWidth;
            int y = m_MapHeight;
            int x2 = 0;
            int y2 = 0;
            foreach (MapObject obj in this.m_Objects)
            {
                if (obj.Region.Left < x) x = obj.Region.Left;
                if (obj.Region.Right > x2) x2 = obj.Region.Right;
                if (obj.Region.Top < y) y = obj.Region.Top;
                if (obj.Region.Bottom > y2) y2 = obj.Region.Bottom;
            }
            if (x2 - x > 0 && y2 - y > 0)
            {
                return new Rectangle(x, y, x2 - x, y2 - y);
            }
            else return Rectangle.Empty;
        }

        public Rectangle GetOuterContentBorder(double rmargin_left, double rmargin_top, double rmargin_right, double rmargin_bottom, int min_cross_dim)
        {
            return GetOuterContentBorder(
             (int)(rmargin_left * m_MapWidth),
             (int)(rmargin_top * m_MapHeight),
             (int)(rmargin_right * m_MapWidth),
             (int)(rmargin_bottom * m_MapHeight),
             min_cross_dim);
        }

        public Rectangle GetOuterContentBorder(int margin_left, int margin_top, int margin_right, int margin_bottom, int min_cross_dim)
        {
            int x = m_MapWidth;
            int y = m_MapHeight;
            int x2 = 0;
            int y2 = 0;
            foreach (MapObject obj in this.m_Objects)
            {
                if (((obj.Region.Left < margin_left) || (obj.Region.Right < margin_left)) ||
                   ((obj.Region.Right > m_MapWidth - margin_right) || (obj.Region.Left > m_MapWidth - margin_right)) ||
                   ((obj.Region.Top < margin_top) || (obj.Region.Bottom < margin_top)) ||
                   ((obj.Region.Bottom > m_MapHeight - margin_bottom) || (obj.Region.Top > m_MapHeight - margin_bottom)))
                {
                    // object crossing margin 
                    continue;
                }

                if (min_cross_dim == 0 || (obj.Region.Width > min_cross_dim && obj.Region.Height > min_cross_dim))
                {
                    // ok object inside
                    if (obj.Region.Left < x) x = obj.Region.Left;
                    if (obj.Region.Right > x2) x2 = obj.Region.Right;
                    if (obj.Region.Top < y) y = obj.Region.Top;
                    if (obj.Region.Bottom > y2) y2 = obj.Region.Bottom;
                }
            }
            if (x2 - x > 0 && y2 - y > 0)
            {
                return new Rectangle(x, y, x2 - x, y2 - y);
            }
            else return Rectangle.Empty;
        }

        #endregion

        #region Get / Remove Objects Inside / Outside Region

        /// <summary>
        /// Remove the objects in the map outside the region, 
        /// if you want to remove object that lie partially inside/outside then
        /// enable remove crossing
        /// </summary>
        /// <param name="rcRegion">the region to scan in</param>
        /// <param name="bRemoveCrossing">also remove objects partially outside the region</param>
        public void RemoveObjectsOutside(Rectangle rcRegion, bool bRemoveCrossing)
        {
            RECT rc = rcRegion.ToRECT();

            List<MapObject> lToRemove = new List<MapObject>();
            foreach (MapObject p in m_Objects)
            {
                if (rc.Contains(p.Region))
                {
                    // OK 
                }
                else
                {
                    if (bRemoveCrossing)
                    {
                        lToRemove.Add(p);
                    }
                    else
                    {
                        if (rc.OverlapsWith(p.Region)) lToRemove.Add(p);
                    }
                }
            }
            foreach (MapObject p in lToRemove)
            {
                ReplaceMap(p, 0);
                m_Objects.Remove(p);
            }
        }

        public void RemoveObjects(IEnumerable<MapObject> objects)
        {
            foreach (MapObject p in objects)
            {
                ReplaceMap(p, 0);
                m_Objects.Remove(p);
            }
        }

        /// <summary>
        /// Remove all objects that touch the page boundary 
        /// the margins are absolute cooordinates, not offsets!
        /// </summary>
        public void RemoveBorderObjects(int margin_left, int margin_top, int margin_right, int margin_bottom)
        {
            int d = 0;

            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    for (int i = 0; i < m_Objects.Count; i++)
                    {
                        MapObject obj = m_Objects[i];
                        if (obj == null) continue;

                        int x = obj.Region.Left;
                        int x2 = obj.Region.Right;
                        int y = obj.Region.Top;
                        int y2 = obj.Region.Bottom;

                        if (((x < margin_left) || (x2 < margin_left)) ||
                           ((x2 > margin_right) || (x > margin_right)) ||
                           ((y < margin_top) || (y2 < margin_top)) ||
                           ((y2 > margin_bottom) || (y > margin_bottom)))
                        {
                            ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, obj, 0, null);
                            m_Objects[i] = null;
                            d++;
                            continue;
                        }
                    }
                }
            }
            if (d > 0) m_Objects.RemoveAll(pRemoveIfNULL);
        }

        public MapObjectList GetObjectsInside(Rectangle rcRegion, bool bIncludePartial)
        {
            return GetObjectsInside(m_Objects, rcRegion, bIncludePartial);
        }

        public MapObjectList GetObjectsInside(IEnumerable<MapObject> objects, Rectangle rcRegion, bool bIncludePartial)
        {
            RECT rc = rcRegion.ToRECT();

            MapObjectList l = new MapObjectList();
            foreach (MapObject p in objects)
            {
                if (rc.Contains(p.Region))
                {
                    // object completely inside rc
                    l.Add(p);
                }
                else
                 if (bIncludePartial)
                {
                    if (rc.OverlapsWith(p.Region)) l.Add(p);
                }
            }
            return l;
        }

        #endregion

        #region Remove Single Pixel Objects

        protected bool pRemoveIfNULL(MapObject obj)
        {
            return obj == null;
        }

        public void RemoveProbableDistortion()
        {
            int d = 0;

            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    for (int i = 0; i < m_Objects.Count; i++)
                    {
                        MapObject obj = m_Objects[i];
                        if (obj == null) continue;

                        // remove single pixels 
                        if (obj.Density == 1)
                        {
                            ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, obj, 0, null);
                            m_Objects[i] = null;
                            d++;
                            continue;
                        }

                        if (obj.Density <= 4)
                        {
                            // also until 3 (TODO dpi dependant) pixels if they are diagonal: surface >= density^2
                            // a straight line wont be deleted this way: surface = 2x1 pixels, line of 2 pixels =>
                            // 2 >= 2^2 == false
                            if (obj.Region.Surface >= (obj.Density * obj.Density) - 1)
                            {
                                ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, obj, 0, null);
                                m_Objects[i] = null;
                                d++;
                                continue;
                            }
                            else
                            {
                                // when not close to a larger object.. 
                            }
                        }
                    }
                }
            }
            if (d > 0) m_Objects.RemoveAll(pRemoveIfNULL);
        }

        #endregion

        #region Map Tracers

        public unsafe float ScanSlopeVertical(uint* pMap, uint object_id, int start_y, int end_y, int start_x, int end_x)
        {
            if (pMap == null) return -1;

            // take 8 samples from the domain(startx, endx) discarding border pieces
            // -> domain divided in 10 pieces, using the middle 8 

            int* nx = stackalloc int[8];
            int* ny = stackalloc int[8];

            int y_dir;
            if (end_y > start_y) y_dir = 1; else y_dir = -1;
            int map_inc = y_dir * m_MapWidth;

            double wdiv10 = (double)(end_x > start_x ? (end_x - start_x) : (start_x - end_x)) / 10.0;
            if (wdiv10 == 0) return 0;

            // the x / y loops should be the other way around.. more cache hits!!
            for (int n = 0; n < 8; n++)
            {
                int x = ((start_x > end_x) ? end_x : start_x) + (int)(wdiv10 * (n + 1));

                int y = -1;
                int xpp = 0;

                while (y != end_y && xpp < 5)
                {
                    uint* pScan = pMap + x + xpp + (start_y * m_MapWidth);

                    for (y = start_y; y != end_y; y += y_dir)
                    {
                        if (*pScan == object_id)
                        {
                            nx[n] = x + xpp;
                            ny[n] = y;
                            xpp = 6;
                            break;
                        }
                        pScan += map_inc;
                    }

                    // maybe we hit a hole, check next x
                    xpp++;
                }
            }

            // calculate the (averaged) slope from the calculation of points
            int* nxy = stackalloc int[8];
            for (int i = 0; i < 8; i++) nxy[i] = nx[i] * ny[i];

            int* nx2 = stackalloc int[8];
            for (int i = 0; i < 8; i++) nx2[i] = nx[i] * nx[i];

            int sum_x = 0;
            int sum_y = 0;
            int sum_xy = 0;
            int sum_x2 = 0;

            for (int i = 0; i < 8; i++)
            {
                sum_x += nx[i];
                sum_y += ny[i];
                sum_xy += nxy[i];
                sum_x2 += nx2[i];
            }

            if (sum_xy == 0) return 0f;

            float slope = (8f * sum_xy - sum_x * sum_y) / (8f * sum_x2 - sum_x * sum_x);
            if ((slope == 0) || (double.IsNaN(slope))) return 0f;

            float angle = (180F * MathF.Atan(slope)) / MathF.PI;

            if (slope > 0) angle = 180f - angle;

            if (angle.InMargin(0, -0.05f, +0.05f)) angle = 0f;

            return angle;
        }

        public unsafe float ScanSlopeHorizontal(uint* pMap, uint object_id, int start_y, int end_y, int start_x, int end_x)
        {
            if (pMap == null) return -1;

            int* nx = stackalloc int[8];
            int* ny = stackalloc int[8];

            int x_dir;
            if (end_x > start_x) x_dir = 1; else x_dir = -1;

            float hdiv10 = (float)(end_y > start_y ? (end_y - start_y) : (start_y - end_y)) / 10f;
            if (hdiv10 == 0) return 0f;

            for (int n = 0; n < 8; n++)
            {
                int y = ((start_y > end_y) ? end_y : start_y) + (int)(hdiv10 * (n + 1));

                int x = start_x;
                int ypp = 0;

                while (x != end_x && ypp < 5)
                {
                    uint* pScan = pMap + ((y + ypp) * m_MapWidth) + start_x;

                    for (x = start_x; x != end_x; x += x_dir)
                    {
                        if (*pScan == object_id)
                        {
                            nx[n] = x;
                            ny[n] = y;
                            ypp = 6;
                            break;
                        }
                        pScan += x_dir;
                    }

                    ypp++;
                }
            }

            // calculate the (averaged) slope from the collection of points using linear regression
            int* nxy = stackalloc int[8];
            for (int i = 0; i < 8; i++) nxy[i] = nx[i] * ny[i];

            int* nx2 = stackalloc int[8];
            for (int i = 0; i < 8; i++) nx2[i] = nx[i] * nx[i];

            int sum_x = 0;
            int sum_y = 0;
            int sum_xy = 0;
            int sum_x2 = 0;

            for (int i = 0; i < 8; i++)
            {
                sum_x += nx[i];
                sum_y += ny[i];
                sum_xy += nxy[i];
                sum_x2 += nx2[i];
            }

            if (sum_xy == 0) return 0f;

            float slope = (8f * sum_xy - sum_x * sum_y) / (8f * sum_x2 - sum_x * sum_x);
            if ((slope == 0) || (float.IsNaN(slope))) return 0f;

            // calculate the angle from the slope, were looking horizontally for a vertical border
            // so when there is no rotation, the angle = 90 degrees
            float angle = 90f - ((180f * MathF.Atan(slope)) / MathF.PI);
            if (angle.InMargin(0, -0.05f, +0.05f)) angle = 0f;

            return angle;
        }

        public QUAD ScanQuad(uint objectid, RECT region)
        {
            unsafe
            {
                fixed (uint* p = Map)
                {
                    return ScanQuad(p, objectid, region);
                }
            }
        }


        public unsafe QUAD ScanQuad(uint* pmap, uint objectid, RECT region)
        {
            // determine center of frame from map points 
            uint* p = pmap + region.Left + region.Top * m_MapWidth;
            int c = 0;
            int cx = 0;
            int cy = 0;

            for (int iy = 0; iy < region.Height; iy++)
            {
                for (int ix = 0; ix < region.Width; ix++)
                {
                    if (*p++ == objectid)
                    {
                        cx += ix;
                        cy += iy;
                        c++;
                    }
                }
                p += m_MapWidth - region.Width;
            }
            cx = (cx / c);
            cy = (cy / c);

            // run again maintaining the points the furthest from the center in kwadrants
            int x1, x2, x3, x4, y1, y2, y3, y4;

            GetQuadrant(pmap, objectid, region.Left, region.Top, cx, cy, out x1, out y1, 0, 0, region.Width / 2, region.Height / 2);
            GetQuadrant(pmap, objectid, region.Left, region.Top, cx, cy, out x2, out y2, region.Width / 2, 0, region.Width, region.Height / 2);
            GetQuadrant(pmap, objectid, region.Left, region.Top, cx, cy, out x3, out y3, 0, region.Height / 2, region.Width / 2, region.Height);
            GetQuadrant(pmap, objectid, region.Left, region.Top, cx, cy, out x4, out y4, region.Width / 2, region.Height / 2, region.Width, region.Height);

            return new QUAD(
             x1 + region.Left, y1 + region.Top,
             x2 + region.Left, y2 + region.Top,
             x3 + region.Left, y3 + region.Top,
             x4 + region.Left, y4 + region.Top);
        }

        unsafe private void GetQuadrant(uint* pmap, uint objectid, int ox, int oy, int cx, int cy, out int xout, out int yout, int x1, int y1, int x2, int y2)
        {
            uint* p = pmap + ox + oy * m_MapWidth;
            int md = 0;
            xout = cx;
            yout = cy;
            for (int iy = y1; iy < y2; iy++)
                for (int ix = x1; ix < x2; ix++)
                    if (*(p + ix + iy * m_MapWidth) == objectid)
                    {
                        int d = (cx - ix) * (cx - ix) + (cy - iy) * (cy - iy);
                        if (d > md)
                        {
                            md = d;
                            xout = ix;
                            yout = iy;
                        }
                    }
        }


        /// <summary>
        /// Geometric mean distance scan vertical from border
        /// Unsafe code, allocating on the stack!!
        /// </summary>
        /// <returns>the geometric mean of the distance of the content from the object border, -1 for failures</returns>
        public unsafe int GMScanVertical(uint* pMap, uint object_id, int start_y, int end_y, int start_x, int end_x)
        {
            if (pMap == null) return -1;

            int n = end_x - start_x;
            if (n <= 0) return -1;

            int* pout = stackalloc int[n];

            int y_dir;
            if (end_y > start_y) y_dir = 1; else y_dir = -1;
            int map_inc = y_dir * m_MapWidth;

            n = 0;
            // the x / y loops should be the other way around.. more cache hits!!
            for (int x = start_x; x <= end_x; x++)
            {
                uint* pScan = pMap + x;
                int* p = pout + n;
                int y;

                for (y = start_y; y < end_y; y += y_dir)
                {
                    if (*pScan == object_id)
                    {
                        *p = y - start_y;
                        break;
                    }
                    pScan += map_inc;
                }

                if (y == end_y) *p = -1;

                n++;
            }

            // calc the geometric mean: the nth root of n numbers multiplied 
            int n1 = Math.Min(2, (int)(0.05 * n));
            int n2 = n - n1;
            if (n2 - n1 <= 0)
            {
                n1 = 0;
                n2 = n;
            }

            int c = 0;
            int t = 1;

            for (n = n1; n < n2; n++)
            {
                int i = pout[n] + 1;
                if (i < 0) continue;

                c++;

                if (t == 0) t = i;
                else t = t * i;
            }

            // take the nth root, and return it. 
            t = (int)Math.Pow(t, 1 / c);

            return t;
        }

        /// <summary>
        /// Geometric mean distance scan vertical from border
        /// Unsafe code, allocating on the stack!!
        /// </summary>
        /// <returns>the geometric mean of the distance of the content from the object border, -1 for failures</returns>
        public unsafe int GMScanHorizontal(uint* pMap, uint object_id, int start_x, int end_x, int start_y, int end_y)
        {
            if (pMap == null) return -1;

            int n = end_y - start_y;
            if (n <= 0) return -1;

            int* pout = stackalloc int[n];

            int x_dir;
            if (end_x > start_x) x_dir = 1; else x_dir = -1;

            n = 0;
            uint* pScan = pMap + start_y * m_MapWidth + start_x;

            for (int y = start_y; y <= end_y; y++)
            {
                int* p = pout + n;
                uint* pSample = pScan;
                int x;

                for (x = start_x; x < end_x; x += x_dir)
                {
                    if (*pSample == object_id)
                    {
                        *p = y - start_y;
                        break;
                    }

                    pSample += x_dir;
                }

                pScan += m_MapWidth;
                if (x == end_x) *p = -1;

                n++;
            }

            // calc the geometric mean: the nth root of n numbers multiplied 
            // this is the same code as the vertical scan as stack is used 
            int n1 = Math.Min(2, (int)(0.05 * n));
            int n2 = n - n1;
            if (n2 - n1 <= 0)
            {
                n1 = 0;
                n2 = n;
            }

            int c = 0;
            int t = 1;

            for (n = n1; n < n2; n++)
            {
                int i = pout[n] + 1;
                if (i < 0) continue;

                c++;

                if (t == 0) t = i;
                else t = t * i;
            }

            // take the nth root, and return it. 
            t = (int)Math.Pow(t, 1 / c);

            return t;
        }


        public static unsafe int TraceVertical(uint* pMap, int map_width, uint object_id, int x, int start_y, int end_y)
        {
            if (pMap == null) return -1;

            uint* pLine = pMap + start_y * map_width + x;
            if (start_y < end_y)
            {
                for (int y = start_y; y <= end_y; y++)
                {
                    if (*pLine == object_id) return y;
                    pLine += map_width;
                }
            }
            else
            {
                for (int y = start_y; y >= end_y; y--)
                {
                    if (*pLine == object_id) return y;
                    pLine -= map_width;
                }
            }

            return end_y;
        }

        public static unsafe int TraceVertical(uint* pMap, int map_width, int x, int start_y, int end_y)
        {
            if (pMap == null) return -1;

            uint* pLine = pMap + start_y * map_width + x;
            if (start_y < end_y)
            {
                for (int y = start_y; y <= end_y; y++)
                {
                    if (*pLine > 0) return y;
                    pLine += map_width;
                }
            }
            else
            {
                for (int y = start_y; y >= end_y; y--)
                {
                    if (*pLine > 0) return y;
                    pLine -= map_width;
                }
            }

            return end_y;
        }

        public int TraceVertical(uint object_id, int x, int start_y, int end_y)
        {
            if (m_Map == null) return -1;

            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    return ObjectMap.TraceVertical(pFixedMap, m_MapWidth, object_id, x, start_y, end_y);
                }
            }
        }

        public static unsafe int TraceHorizontal(uint* pMap, int map_width, uint object_id, int y, int start_x, int end_x)
        {
            if (pMap == null) return -1;
            uint* pLine = pMap + y * map_width;

            if (start_x < end_x)
            {
                pLine += start_x;
                for (int x = start_x; x <= end_x; x++)
                {
                    if (*pLine == object_id) return x;
                    pLine++;
                }
            }
            else
            {
                pLine += start_x;
                for (int x = start_x; x >= end_x; x--)
                {
                    if (*pLine == object_id) return x;
                    pLine--;
                }
            }

            return end_x;
        }

        public static unsafe int TraceHorizontal(uint* pMap, int map_width, int y, int start_x, int end_x)
        {
            if (pMap == null) return -1;
            uint* pLine = pMap + y * map_width;

            if (start_x < end_x)
            {
                pLine += start_x;
                for (int x = start_x; x <= end_x; x++)
                {
                    if (*pLine > 0) return x;
                    pLine++;
                }
            }
            else
            {
                pLine += start_x;
                for (int x = start_x; x >= end_x; x--)
                {
                    if (*pLine > 0) return x;
                    pLine--;
                }
            }

            return end_x;
        }

        public int TraceHorizontal(uint object_id, int y, int start_x, int end_x)
        {
            if (m_Map == null) return -1;
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    return ObjectMap.TraceHorizontal(pFixedMap, m_MapWidth, object_id, y, start_x, end_x);
                }
            }
        }

        /// <summary>
        /// Trace the outerborder for an object found in the map, 
        /// the rectangle returned is in map coord space
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public RECT TraceOuterBorder(MapObject obj)
        {
            if (m_Map == null) return RECT.Invalid;
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    return TraceOuterBorder(obj, pFixedMap);
                }
            }
        }

        public unsafe RECT TraceOuterBorder(MapObject obj, uint* pMap)
        {
            if (obj == null) return RECT.Invalid;
            if (pMap == null) return RECT.Invalid;

            int x = obj.Region.Left;
            int y = obj.Region.Top;

            RECT rc = new RECT(
             obj.Region.Right,
             obj.Region.Bottom,
             obj.Region.Left,
             obj.Region.Top
            );

            while (y <= obj.Region.Bottom)
            {
                int t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, obj.Region.Left, rc.Left);

                if (t == obj.Region.Right)
                {
                    // no pixels in scanline.. left & right unchanged, only applies for top margin
                }
                else
                {
                    rc.Left = Math.Min(rc.Left, t);
                    t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, obj.Region.Right, rc.Right);
                    rc.Right = Math.Max(rc.Right, t);
                }
                y++;
            }

            // limit the domain of x with the result from the horizontal scans
            x = rc.Left;
            while (x <= rc.Right)
            {
                rc.Top = Math.Min(rc.Top, ObjectMap.TraceVertical(pMap, m_MapWidth, obj.Id, x, obj.Region.Top, rc.Top));
                rc.Bottom = Math.Max(rc.Bottom, ObjectMap.TraceVertical(pMap, m_MapWidth, obj.Id, x, obj.Region.Bottom, rc.Bottom));
                x++;
            }

            if (rc.Left > rc.Right || rc.Top > rc.Bottom || rc.Left == rc.Right || rc.Top == rc.Bottom)
            {
                rc.Left = -1;
                rc.Right = -1;
                rc.Top = -1;
                rc.Right = -1;
            }

            return rc;
        }

        public RECT TraceInnerBorder(MapObject obj)
        {
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    return TraceInnerBorder(obj, pFixedMap);
                }
            }
        }

        public unsafe RECT TraceInnerBorder(MapObject obj, uint* pMap)
        {
            // grow from the center. 
            int cx = (obj.Region.Left + obj.Region.Right) / 2;
            int cy = (obj.Region.Top + obj.Region.Bottom) / 2;

            int ox1 = Math.Min(obj.Region.Left, obj.Region.Right);
            int oy1 = Math.Min(obj.Region.Top, obj.Region.Bottom);
            int ox2 = Math.Max(obj.Region.Left, obj.Region.Right);
            int oy2 = Math.Max(obj.Region.Top, obj.Region.Bottom);

            // get inner h from center
            int y1 = ObjectMap.TraceVertical(pMap, m_MapWidth, obj.Id, cx, cy, oy1);
            int y2 = ObjectMap.TraceVertical(pMap, m_MapWidth, obj.Id, cx, cy, oy2);

            // if == center, object has stuf in center.. 
            // i could use another method that grows from holes in the object and 
            // then returns the largest..
            if ((y1 == cy) || (y2 == cy)) return RECT.Invalid;

            // grow upward, scanning <> x
            int y = cy;
            int x1y1 = ox1;
            int x2y1 = ox2;

            while (y > y1)
            {
                int t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, cx, x1y1);
                if (t > x1y1)
                {
                    // dont allow the surface to get smaller!
                    if ((x2y1 - x1y1) * (cy - y - 1) > (x2y1 - t) * (cy - y))
                    {
                        ++y;
                        break;
                    }
                    x1y1 = t;
                }

                t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, cx, x2y1);
                if (t < x2y1)
                {
                    if ((x2y1 - x1y1) * (cy - y - 1) > (t - x1y1) * (cy - y))
                    {
                        ++y;
                        break;
                    }
                    x2y1 = t;
                }
                y--;
            }
            y1 = y;

            // then down 
            y = cy;
            int x1y2 = x1y1;
            int x2y2 = x2y1;
            while (y < y2 - 1)
            {
                int t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, cx, x1y2);
                if (t > x1y2)
                {
                    if ((x2y2 - x1y2) * (y - cy - 1) > (x2y2 - t) * (y - cy))
                    {
                        --y;
                        break;
                    }
                    x1y2 = t;
                }

                t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, obj.Id, y, cx, x2y2);
                if (t < x2y2)
                {
                    if ((x2y2 - x1y2) * (y - cy - 1) > (t - x1y2) * (y - cy))
                    {
                        --y;
                        break;
                    }
                    x2y2 = t;
                }

                y++;
            }
            y2 = y;

            // we now have a possibly skewed rectangle in xn/yn
            // for now we take the min sides
            // TODO
            // better would be to take the one with the largest surface, that possibly requires
            // maintaining a list of 1<x>2 
            return new RECT(
             Math.Max(x1y1, x1y2) + 1,
             y1 + 1,
             Math.Min(x2y1, x2y2) - 1,
             y2 - 1
             );
        }



        public unsafe RECT TraceInnerBorderAny(MapObject obj, uint* pMap)
        {
            // grow from the center. 
            int cx = (obj.Region.Left + obj.Region.Right) / 2;
            int cy = (obj.Region.Top + obj.Region.Bottom) / 2;

            int ox1 = Math.Min(obj.Region.Left, obj.Region.Right);
            int oy1 = Math.Min(obj.Region.Top, obj.Region.Bottom);
            int ox2 = Math.Max(obj.Region.Left, obj.Region.Right);
            int oy2 = Math.Max(obj.Region.Top, obj.Region.Bottom);

            // get inner h from center
            int y1 = ObjectMap.TraceVertical(pMap, m_MapWidth, cx, cy, oy1);
            int y2 = ObjectMap.TraceVertical(pMap, m_MapWidth, cx, cy, oy2);

            // if == center, object has stuf in center.. 
            // i could use another method that grows from holes in the object and 
            // then returns the largest..
            if ((y1 == cy) || (y2 == cy)) return RECT.Invalid;

            // grow upward, scanning <> x
            int y = cy;
            int x1y1 = ox1;
            int x2y1 = ox2;

            while (y > y1)
            {
                int t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, y, cx, x1y1);
                if (t > x1y1)
                {
                    // dont allow the surface to get smaller!
                    if ((x2y1 - x1y1) * (cy - y - 1) > (x2y1 - t) * (cy - y))
                    {
                        ++y;
                        break;
                    }
                    x1y1 = t;
                }

                t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, y, cx, x2y1);
                if (t < x2y1)
                {
                    if ((x2y1 - x1y1) * (cy - y - 1) > (t - x1y1) * (cy - y))
                    {
                        ++y;
                        break;
                    }
                    x2y1 = t;
                }
                y--;
            }
            y1 = y;

            // then down 
            y = cy;
            int x1y2 = x1y1;
            int x2y2 = x2y1;
            while (y < y2 - 1)
            {
                int t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, y, cx, x1y2);
                if (t > x1y2)
                {
                    if ((x2y2 - x1y2) * (y - cy - 1) > (x2y2 - t) * (y - cy))
                    {
                        --y;
                        break;
                    }
                    x1y2 = t;
                }

                t = ObjectMap.TraceHorizontal(pMap, m_MapWidth, y, cx, x2y2);
                if (t < x2y2)
                {
                    if ((x2y2 - x1y2) * (y - cy - 1) > (t - x1y2) * (y - cy))
                    {
                        --y;
                        break;
                    }
                    x2y2 = t;
                }

                y++;
            }
            y2 = y;

            // we now have a possibly skewed rectangle in xn/yn
            // for now we take the min sides
            // TODO
            // better would be to take the one with the largest surface, that possibly requires
            // maintaining a list of 1<x>2 
            return new RECT(
             Math.Max(x1y1, x1y2) + 1,
             y1 + 1,
             Math.Min(x2y1, x2y2) - 1,
             y2 - 1
             );
        }


        public bool ScanStartEnd(int y, int min_object_size, out int start, out int end)
        {
            start = -1;
            end = -1;
            if (m_Map == null) return false;
            unsafe
            {
                fixed (uint* pSample = m_Map)
                {
                    int c = 0;
                    uint* scan = pSample + (y * m_MapWidth);
                    for (int x = 0; x < m_MapWidth; x++)
                    {
                        if (*scan++ > 0)
                        {
                            c++;
                            if (c > min_object_size)
                            {
                                start = x - c;
                                break;
                            }
                        }
                        else c = 0;
                    }
                    if (start > 0)
                    {
                        scan = pSample + (y * m_MapWidth) + m_MapWidth;
                        c = 0;
                        for (int x = m_MapWidth; x >= 0; --x)
                        {
                            if (*--scan > 0)
                            {
                                c++;
                                if (c > min_object_size)
                                {
                                    end = x + c;
                                    return true;
                                }
                            }
                            else c = 0;
                        }
                    }
                }
            }
            return false;
        }

        #endregion

        #region Image Generation

        /// <summary>
        /// Generate a bitmap from the mapdata
        /// </summary>
        /// <returns></returns>
        public Bitmap GenerateImage()
        {
            return GenerateImage(null, this.Objects);
        }

        public Bitmap GenerateImage(IEnumerable<MapObject> to_draw)
        {
            return GenerateImage(null, to_draw);
        }

        public Bitmap GenerateImage(MapObjectType[] types)
        {
            return GenerateImage(types, this.Objects);
        }

        /// <summary>
        /// generate bitmap from mapdata using only the data from given types
        /// </summary>
        /// <param name="types"></param>
        /// <returns></returns>
        public Bitmap GenerateImage(MapObjectType[] types, IEnumerable<MapObject> to_draw)
        {
            if ((m_Map == null) || (m_MapHeight == 0) || (m_MapWidth == 0)) return null;

            Bitmap bmp = new Bitmap(m_MapWidth, m_MapHeight, PixelFormat.Format24bppRgb);

            Random r = new Random();

            SolidBrush bText = new SolidBrush(Color.Blue);

            bool bFrame = false;

            // fill with white
            using (Graphics gfx = Graphics.FromImage(bmp))
            {
                gfx.FillRectangle(
                 new SolidBrush(Color.White),
                 new Rectangle(0, 0, m_MapWidth, m_MapHeight)
                );

                // Fill in objects 
                foreach (MapObject obj in to_draw)
                {
                    // SolidBrush rsb = new SolidBrush(Color.FromArgb(255, 200 + r.Next(30), 200 + r.Next(30), 200 + r.Next(55)));

                    if (types != null && types.Length > 0)
                        if (!types.Contains(obj.Type)) continue;

                    Color c;

                    switch (obj.Type)
                    {

                        case MapObjectType.Frame:
                            {
                                c = Color.Red;
                                break;
                            }
                        case MapObjectType.Solid:
                            {
                                c = Color.Blue;
                                break;
                            }
                        case MapObjectType.LogoPart:
                            {
                                c = Color.Cyan;
                                break;
                            }

                        case MapObjectType.Text:
                            {
                                c = Color.Green;
                                break;
                            }

                        default:
                        case MapObjectType.Other:
                            {
                                c = Color.Black;
                                break;
                            }
                    }

                    if (obj.Type != MapObjectType.Frame)
                    {
                        //   gfx.FillRectangle(rsb, new Rectangle(obj.Region.Left, obj.Region.Top, obj.Region.Width, obj.Region.Height));
                    }
                    else
                    {
                        if (obj.ChildCount > 0)
                        {
                            foreach (MapObject o2 in obj.Children)
                            {
                                if (o2.Type == MapObjectType.Text)
                                {
                                    //    gfx.FillRectangle(rsb, new Rectangle(o2.Region.Left, o2.Region.Top, o2.Region.Width, o2.Region.Height));
                                }
                            }
                        }
                    }

                    /*if (obj.ChildCount > 0)
                    {
                     foreach (MapObject child in obj.Children)
                     {
                      gfx.DrawRectangle(Pens.Black, new Rectangle(child.Region.Left, child.Region.Top, child.Region.Width, child.Region.Height));
                     } 
                    } */

                    for (int y = obj.Region.Top; y <= obj.Region.Bottom; y++)
                    {
                        for (int x = obj.Region.Left; x <= obj.Region.Right; x++)
                        {
                            if (m_Map[y * m_MapWidth + x] > 0)
                            {
                                bmp.SetPixel(x, y, c);
                            }
                        }
                    }

                    if (obj.Type == MapObjectType.Frame && !bFrame)
                    {
                        //bFrame = true;
                        MapFrameObject frame = (MapFrameObject)obj;

                        RECT rcInner = TraceInnerBorder(obj);
                        gfx.DrawRectangle(Pens.Blue, rcInner.Rectangle);

                        // draw slope angles 
                        float fx = Math.Min(obj.Region.Left, obj.Region.Right) + 40;
                        float fy = Math.Min(obj.Region.Top, obj.Region.Bottom) + 10;
                        gfx.DrawString("Left: " + frame.LeftSlope.ToString("0.00"), SystemFonts.DefaultFont, bText, fx, fy);
                        fy += 15;
                        gfx.DrawString("Top: " + frame.TopSlope.ToString("0.00"), SystemFonts.DefaultFont, bText, fx, fy);
                        fy += 15;
                        gfx.DrawString("Right: " + frame.RightSlope.ToString("0.00"), SystemFonts.DefaultFont, bText, fx, fy);
                        fy += 15;
                        gfx.DrawString("Bottom: " + frame.BottomSlope.ToString("0.00"), SystemFonts.DefaultFont, bText, fx, fy);


                    }

                }
            }

            return bmp;
        }


        #endregion

        #region Classification

        protected Distribution m_Density = new Distribution();
        protected Distribution m_Area = new Distribution();
        protected double m_AverageTextWidth = 0.0;
        protected double m_AverageTextHeight = 0.0;

        /// <summary>
        /// distribution of object area
        /// </summary>
        public Distribution Area
        {
            get
            {
                return m_Area;
            }
        }

        /// <summary>
        /// distribution of object density
        /// </summary>
        public Distribution Density
        {
            get
            {
                return m_Density;
            }
        }

        /// <summary>
        /// Calc distribution of object properties like area and density
        /// </summary>
        protected void CalculateDistributions()
        {
            m_Area.Clear();
            m_Density.Clear();
            foreach (MapObject obj in m_Objects)
            {
                if (obj == null) continue;
                m_Density.AddSample(obj.Density);
                m_Area.AddSample((uint)obj.Region.Surface);
            }
            m_Area.Calculate(1.2, 0.8);
            m_Density.Calculate(1.2, 0.8);
        }


        /// <summary>
        /// classify the objects found into 4 groups:
        /// 
        /// - SOLID
        /// - FRAME
        /// - TEXT
        /// - OTHER
        /// </summary>
        public void ClassifyObjects()
        {
            if (m_Map == null) throw new ArgumentNullException("m_Map == null");
            if ((m_Objects == null) || (m_Objects.Count == 0)) return;

            RemoveProbableDistortion();
            RemoveBorderObjects(2, 2, this.m_MapWidth - 2, this.m_MapHeight - 2);

            CalculateDistributions();

            // 
            uint cFrame = 0;
            uint cText = 0;
            uint cSolid = 0;
            uint cOther = 0;
            m_AverageTextWidth = 0;
            m_AverageTextHeight = 0;
            uint m_MaxTextWidth = (uint)(0.01 * m_MapWidth);
            uint m_MaxTextHeight = (uint)(0.03 * m_MapHeight);

            //
            for (int i = 0; i < m_Objects.Count; i++)
            {
                MapObject obj = m_Objects[i];
                if (obj == null) continue;

                double dArea = 1.0 / m_Area.MedianAverage;
                double d = (double)obj.Density / obj.Region.Surface;

                double s = dArea * obj.Region.Surface;
                double w = obj.Region.Width;
                double h = obj.Region.Height;


                // Classify on solidity 
                // - > 90%     -high chance of rectangle/line
                //             >> Note: this may also apply to an i . l or , depending on font

                //                      a sans serif font will have an l an i that resemble a rectangle
                // - 10%..90%  -mostly text
                // - < 10%     -higly probable that this is a frame
                //               to be one it:
                //               - should be larger then the avg text size,

                // SOLID
                if (obj.Density > 0.92 * obj.Region.Surface)
                {
                    cSolid++;
                    m_Objects[i] = (MapObject)MapSolidObject.FromObject(obj);
                    continue;
                }

                // FRAME
                if (IsFrame(obj))
                {
                    cFrame++;

                    MapFrameObject frame = MapFrameObject.FromObject(obj);
                    m_Objects[i] = (MapObject)frame;

                    unsafe
                    {
                        fixed (uint* pFixedMap = m_Map)
                        {
                            // scan the slope angles of the sides of the frame
                            // the frame can have rounded corners.. adjust for it as that will give false results
                            int c = frame.InnerRegion.Left - frame.Region.Left;
                            c = Math.Max(frame.Region.Right - frame.InnerRegion.Right, c);
                            c = Math.Max(frame.InnerRegion.Top - frame.Region.Top, c);
                            c = Math.Max(frame.Region.Bottom - frame.InnerRegion.Bottom, c);

                            c = (int)(c * 1.2);

                            frame.LeftSlope = ScanSlopeHorizontal(pFixedMap, frame.Id, frame.Region.Top + c, frame.Region.Bottom - c, frame.Region.Left, frame.Region.Right);
                            frame.RightSlope = ScanSlopeHorizontal(pFixedMap, frame.Id, frame.Region.Top + c, frame.Region.Bottom - c, frame.Region.Right, frame.Region.Left);
                            frame.TopSlope = ScanSlopeVertical(pFixedMap, frame.Id, frame.Region.Top, frame.Region.Bottom, frame.Region.Left + c, frame.Region.Right - c);
                            frame.BottomSlope = ScanSlopeVertical(pFixedMap, frame.Id, frame.Region.Bottom, frame.Region.Top, frame.Region.Left + c, frame.Region.Right - c);
                            continue;
                        }
                    }
                }

                // Assume text from here if ...
                if ((h < m_MaxTextHeight) && (w / 2 < h))
                {

                    // ((w < (m_AverageTextWidth * 3)) && (h < (m_AverageTextHeight * 3)) &&

                    //it could be text, ifso, the entropy value with the density can be usefull, need stats.. 

                    // adjust average text dims, limit both to 1.5 of the other, that
                    // way we can worry less about connected text (bad quality scans / low res image)
                    // text mostly connects horizontally, check width first

                    if (w > (h * 1.5)) w = h * 1.5;
                    if (h > (w * 1.5)) h = w * 1.5;

                    // allow textobjects to be connected over the width,
                    // but dont allow them to be higher then the max 
                    if (h < m_MaxTextHeight)
                    {
                        cText++;
                        obj.Type = MapObjectType.Text; // == text

                        m_AverageTextWidth = (m_AverageTextWidth * (cText - 1) + w) / cText;
                        m_AverageTextHeight = (m_AverageTextHeight * (cText - 1) + h) / cText;

                        continue;
                    }
                }

                cOther++;
                obj.Type = MapObjectType.Other;
            }
            m_Objects.RemoveAll(MapObject.IsNull);

            if ((cOther > 0 && (cFrame > 0)) || cFrame > 1)
            {
                MergeFrames();
            }

            GrowTextObjects();
            BuildFrameTree();
            SearchLogo();
        }

        #region Frame Classification

        protected bool IsFrame(MapObject obj)
        {
            if (obj == null) return false;

            int s = obj.Region.Surface;
            int d = (int)obj.Density;

            // if too dense, return, assume > 20% density as too much
            // todo, this is a statistic variable we need to find out .. TODO 
            if (s == d) return false;
            if (d > 0.2 * s) return false;

            if (obj.Region.Surface < (m_AverageTextWidth * m_AverageTextHeight * 8)) return false;

            unsafe
            {
                fixed (uint* pMap = m_Map)
                {
                    // first trace the inner border 
                    RECT rcInner = TraceInnerBorder(obj, pMap);
                    if (RECT.IsNullOrEmpty(rcInner) || RECT.IsInvalid(rcInner)) return false;

                    // it should atleast be 80% of the original object in both dimensions
                    if (rcInner.Width < obj.Region.Width * 0.8) return false;
                    if (rcInner.Height < obj.Region.Height * 0.8) return false;

                    int sb = obj.Region.Surface - rcInner.Surface;

                    obj.InnerRegion = rcInner;

                    return true;
                }
            }
        }

        /// <summary>
        /// merge frames seperated by distortions 
        /// </summary>
        protected void MergeFrames()
        {
            unsafe
            {
                fixed (uint* pFixedMap = m_Map)
                {
                    // should first remove artifacts. the object count of them can be massive
                    // and will make this very slow, quadratic to objectcount.

                    // foreach frame.. 
                    for (int i = 0; i < m_Objects.Count; i++)
                    {
                        MapObject obj = m_Objects[i];
                        if (obj == null) continue;

                        if (obj.Type != MapObjectType.Frame) continue;

                        // then for each other object.. 
                        for (int j = 0; j < m_Objects.Count; j++)
                        {
                            if (j == i) continue;

                            MapObject obj2 = m_Objects[j];
                            if (obj2 == null) continue;
                            switch (obj2.Type)
                            {
                                case MapObjectType.Other:
                                    {
                                        // only continue if the bounding rectangles overlap
                                        if (!obj.Region.OverlapsWith(obj2.Region)) continue;

                                        MapFrameObject frame = (MapFrameObject)obj;
                                        if (!frame.GM_Calculated)
                                        {
                                            // calculate geometric mean distances for the object
                                            frame.GM_Top = GMScanVertical(pFixedMap, obj.Id, obj.Region.Top, obj.Region.Bottom, obj.Region.Left, obj.Region.Right);
                                            frame.GM_Bottom = GMScanVertical(pFixedMap, obj.Id, obj.Region.Bottom, obj.Region.Top, obj.Region.Left, obj.Region.Right);
                                            frame.GM_Left = GMScanHorizontal(pFixedMap, obj.Id, obj.Region.Left, obj.Region.Right, obj.Region.Top, obj.Region.Bottom);
                                            frame.GM_Right = GMScanHorizontal(pFixedMap, obj.Id, obj.Region.Right, obj.Region.Left, obj.Region.Top, obj.Region.Bottom);
                                            frame.GM_Calculated = true;
                                        }

                                        // belong to eachother??
                                        // we could trace the borders and see if the average on an axis is the same
                                        // for example:
                                        // = if its alligned aloong the bottom axes, then scan vertical along the 
                                        // x-axis of both objects, they both should have the same average not counting 
                                        // rounded corners though...
                                        bool b = false;

                                        if (obj.Region.Bottom.InMargin(obj2.Region.Bottom, 2, 2))
                                        {
                                            int g = GMScanVertical(pFixedMap, obj2.Id, obj2.Region.Bottom, obj2.Region.Top, obj2.Region.Left, obj.Region.Right);
                                            b = frame.GM_Bottom.InMargin(g, 1, 1);
                                        }

                                        if (!b && obj.Region.Top.InMargin(obj2.Region.Top, 2, 2))
                                        {
                                            int g = GMScanVertical(pFixedMap, obj2.Id, obj2.Region.Top, obj2.Region.Bottom, obj2.Region.Left, obj2.Region.Right);
                                            b = frame.GM_Top.InMargin(g, 1, 1);
                                        }

                                        if (!b && obj.Region.Left.InMargin(obj2.Region.Left, 2, 2))
                                        {
                                            int g = GMScanHorizontal(pFixedMap, obj2.Id, obj2.Region.Left, obj2.Region.Right, obj2.Region.Top, obj2.Region.Bottom);
                                            b = frame.GM_Left.InMargin(g, 1, 1);
                                        }

                                        if (!b && obj.Region.Right.InMargin(obj2.Region.Right, 2, 2))
                                        {
                                            int g = GMScanHorizontal(pFixedMap, obj2.Id, obj2.Region.Right, obj2.Region.Left, obj2.Region.Top, obj2.Region.Bottom);
                                            b = frame.GM_Right.InMargin(g, 1, 1);
                                        }

                                        if (b)
                                        {
                                            ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, obj2, frame.Id, frame);

                                            frame.GM_Calculated = false; // force recalc
                                            frame.GM_Left = -1;
                                            frame.GM_Right = -1;
                                            frame.GM_Top = -1;
                                            frame.GM_Bottom = -1;

                                            obj2.Density = 0;
                                            obj2.Id = 0;
                                            m_Objects[j] = null;
                                        }

                                        break;
                                    }
                                case MapObjectType.Frame:
                                    {
                                        // if they dont (almost) overlap, continue
                                        if (!obj.Region.OverlapsWith(
                                         new RECT(
                                          obj2.Region.Left - 3,
                                          obj2.Region.Top - 3,
                                          obj2.Region.Right + 3,
                                          obj2.Region.Bottom + 3))) continue;

                                        // if either contains oneother then skip
                                        if (obj.Region.Contains(obj2.Region)) continue;
                                        if (obj2.Region.Contains(obj.Region)) continue;

                                        // if one of the axes is the same 
                                        // then we have a probable match.. 
                                        // - could check if a frame is closed by checking the inner against the outer rect
                                        // - we could use the slope and thickniss of one frame and project/test that one the 
                                        //   other
                                        if ((obj.Region.Left == obj2.Region.Left) ||
                                           (obj.Region.Right == obj2.Region.Right) ||
                                           (obj.Region.Top == obj2.Region.Top) ||
                                           (obj.Region.Bottom == obj2.Region.Bottom))
                                        {
                                            // keep the larger one
                                            MapObject smaller, larger;
                                            if (obj.Region.Surface > obj2.Region.Surface)
                                            {
                                                smaller = obj2;
                                                larger = obj;
                                            }
                                            else
                                            {
                                                smaller = obj;
                                                larger = obj2;
                                            }

                                            ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, smaller, larger.Id, larger);

                                            if (smaller == obj)
                                            {
                                                obj = null;
                                                m_Objects[i] = null;
                                            }
                                            else
                                            {
                                                m_Objects[j] = null;
                                            }
                                        }
                                        break;
                                    }

                                case MapObjectType.Solid:
                                    {
                                        // if they dont (almost) overlap, continue  
                                        // note the larger margin allowing more missing for small sections
                                        // TODO > this should be derived from DPI and rcinner / rcouter gap size
                                        if (!obj.Region.OverlapsWith(
                                         new RECT(
                                          obj2.Region.Left - 5,
                                          obj2.Region.Top - 5,
                                          obj2.Region.Right + 5,
                                          obj2.Region.Bottom + 5))) continue;

                                        // if either contains oneother then skip
                                        if (obj.Region.Contains(obj2.Region)) continue;
                                        if (obj2.Region.Contains(obj.Region)) continue;

                                        bool b = false;

                                        // could be a line that is part of the frame.. 
                                        // TODO should use the space between rcinner en outer.. 
                                        //      for the second check..
                                        if (obj.Region.Left.InMargin(obj2.Region.Left, 2, 2))
                                        {
                                            b = obj.Region.Left.InMargin(obj2.Region.Right, 5, 5);
                                        }
                                        else
                                         if (obj.Region.Right.InMargin(obj2.Region.Right, 2, 2))
                                        {
                                            b = obj.Region.Right.InMargin(obj2.Region.Left, 5, 5);
                                        }
                                        else
                                          if (obj.Region.Top.InMargin(obj2.Region.Top, 2, 2))
                                        {
                                            b = obj.Region.Top.InMargin(obj2.Region.Bottom, 5, 5);
                                        }
                                        else
                                           if (obj.Region.Bottom.InMargin(obj2.Region.Bottom, 2, 2))
                                        {
                                            b = obj.Region.Bottom.InMargin(obj2.Region.Top, 5, 5);
                                        }

                                        if (b)
                                        {
                                            ObjectMap.ReplaceMap(pFixedMap, m_MapWidth, obj2, obj.Id, obj);
                                            m_Objects[j] = null;
                                        }

                                        break;
                                    }
                            }
                            if (obj == null) break;
                        }
                    }
                }
            }

            m_Objects.RemoveAll(MapObject.IsNull);
            return;
        }

        /// <summary>
        /// Rebuilds the map object list in a tree form and uses the frames (if any as nodes)
        /// iow: the objects in frames are removed from the object list and put into the
        /// children list of the framemapobject.
        /// </summary>
        protected void BuildFrameTree()
        {
            List<MapObject> lFrames = new List<MapObject>();

            // extract frame objects from the list 
            foreach (MapObject obj in m_Objects)
                if (obj.Type == MapObjectType.Frame)
                    lFrames.Add(obj);
            if (lFrames.Count == 0) return;

            foreach (MapObject obj in lFrames) m_Objects.Remove(obj);

            // now for each frame, find the objects that are completely contained in them 
            foreach (MapObject frame in lFrames)
            {
                for (int i = 0; i < m_Objects.Count; i++)
                {
                    MapObject obj = m_Objects[i];
                    if (obj == null) continue;

                    if (frame.Region.Contains(obj.Region))
                    {
                        m_Objects[i] = null;
                        if (frame.Children == null) frame.Children = new List<MapObject>();

                        frame.Children.Add(obj);
                    }
                }
            }

            // add the frames back to the general object list and remove the empty entries
            foreach (MapObject obj in lFrames) m_Objects.Add(obj);
            m_Objects.RemoveAll(MapObject.IsNull);
        }

        #endregion

        #region Text Classification

        /// <summary>
        /// Grow & Concatenate text objects over the x-axis
        /// </summary>
        public void GrowTextObjects()
        {
            if (ObjectCount <= 0) return;

            MapObject[] aobj = m_Objects.ToArray();
            RECT rc = RECT.Null;

            int cobjects = m_Objects.Count;
            m_Objects.Clear();

            // the growt factor
            int g = (int)(m_AverageTextWidth * 1.6);
            int gy = (int)(m_AverageTextHeight * 1.6);

            for (int i = 0; i < aobj.Length; i++)
            {
                MapObject obj = aobj[i];
                if (obj == null) continue;

                // frame = no text
                if (obj.Type == MapObjectType.Frame)
                {
                    aobj[i] = null;
                    m_Objects.Add(obj);
                    continue;
                }
                // solid.. can be.. if small
                if (obj.Type == MapObjectType.Solid && obj.Region.Surface > m_AverageTextWidth * m_AverageTextHeight)
                {
                    aobj[i] = null;
                    m_Objects.Add(obj);
                    continue;
                }

                rc.Left = obj.Region.Left;
                rc.Top = obj.Region.Top;
                rc.Right = obj.Region.Right;
                rc.Bottom = obj.Region.Bottom;

                bool bGrowing = true;
                while (bGrowing)
                {
                    bGrowing = false;

                    // only grow on x-axis, dont force on y!
                    rc.Left = rc.Left - g;
                    rc.Right = rc.Right + g;

                    // - if the centre of the new object is below the bottom or above the (top: what if .)
                    for (int j = 0; j < aobj.Length; j++)
                    {
                        if (i == j) continue;
                        MapObject obj2 = aobj[j];
                        if (obj2 == null) continue;
                        if (obj2.Type == MapObjectType.Frame)
                        {
                            m_Objects.Add(obj2);
                            aobj[j] = null;
                            continue;
                        }

                        //if (obj2.Region.OverlapsWith(rc))
                        Rectangle r = Rectangle.Intersect(
                         new Rectangle(rc.Left, rc.Top, rc.Width, rc.Height),
                         new Rectangle(obj2.Region.Left, obj2.Region.Top, obj2.Region.Width, obj2.Region.Height));

                        if (r.Width > 0 && r.Height > 0)
                        {
                            if (obj.Children == null) obj.Children = new List<MapObject>();
                            obj.Children.Add(obj2);

                            // we have grown but to where? 
                            rc.Left = rc.Left + g;
                            rc.Right = rc.Right - g;

                            if (((rc.Left + rc.Right) / 2) < ((obj2.Region.Left + obj2.Region.Right) / 2))
                            {
                                // grow to right
                                rc.Right = Math.Max(rc.Right, obj2.Region.Right);
                            }
                            else
                            {
                                // to the left
                                rc.Left = Math.Min(rc.Left, obj2.Region.Left);
                            }

                            rc.Bottom = Math.Max(rc.Bottom, obj2.Region.Bottom);
                            rc.Top = Math.Min(rc.Top, obj2.Region.Top);

                            aobj[j] = null;

                            bGrowing = true;
                            break;
                        }
                    }

                    if (!bGrowing)
                    {
                        rc.Left = rc.Left + g;
                        rc.Right = rc.Right - g;
                    }
                }

                // has the object grown?
                if (obj.Children != null)
                {
                    // create a new object from it .. 
                    MapObject n = new MapObject();
                    int x1 = int.MaxValue; //obj.Region.Left;
                    int y1 = int.MaxValue; //obj.Region.Top;
                    int x2 = int.MinValue; // obj.Region.Right;
                    int y2 = int.MinValue; // obj.Region.Bottom;

                    n.Children = obj.Children;
                    obj.Children = null;
                    n.Children.Add(obj);

                    n.Density = 0;
                    foreach (MapObject child in n.Children)
                        if (child != null)
                        {
                            child.Parent = n;
                            n.Density += child.Density;

                            x1 = Math.Min(x1, child.Region.Left);
                            x2 = Math.Max(x2, child.Region.Right);
                            y1 = Math.Min(y1, child.Region.Top);
                            y2 = Math.Max(y2, child.Region.Bottom);
                        }

                    n.Region = new RECT(x1, y1, x2, y2);
                    n.Id = (uint)++cobjects;
                    n.Type = MapObjectType.Text;
                    n.Parent = null;

                    m_Objects.Add(n);
                }

                aobj[i] = null;
            }
        }

        #endregion

        #region Logo Classification

        private bool LogoPartCheck1(MapObject obj)
        {
            double min_surface = m_AverageTextHeight * m_AverageTextWidth * 2;

            if ((obj != null) && (obj.Type != MapObjectType.Frame))
            {
                // dont add objects more then 98% solid => filled block -> unlikely a logo
                // __then again, it could be a combination of colors and this object labeler is
                // colorblind.. we could do a color scan on solid objects to dermine if its
                // just a gradient on a surface or something more complex -> possibly still a logo
                // if on the right spot.. 
                // 
                // also: objects should be dense enough not to be a line or circle or something 
                if ((obj.WeightedDensity < 0.98) && (obj.WeightedDensity > 0.20))
                {
                    // it should have a bigger density then the average text object _surface_
                    // and its dimensions individually should be larger then the avg text size
                    if ((obj.Density > min_surface) &&
                        (obj.Region.Width > m_AverageTextWidth * 1.5) &&
                        (obj.Region.Height > m_AverageTextHeight * 1.5))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        public bool SearchLogo()
        {
            if (ObjectCount == 0) return false;
            List<MapObject> l = new List<MapObject>();

            // sort on surface size
            m_Objects.Sort(MapObject.CompareOnSurface);

            // then get the max? 10? largest objects in a suitable position
            int i = m_Objects.Count - 1;
            int n = Math.Min(20, m_Objects.Count);
            while ((n > 0) && (i >= 0))
            {
                MapObject obj = m_Objects[i];

                // if it is a compound text object
                if ((obj.ChildCount > 0) && (obj.Type == MapObjectType.Text))
                {
                    // then when a!! child could be considered a part of the logo
                    foreach (MapObject child in obj.Children)
                    {
                        if (LogoPartCheck1(child))
                        {
                            // the compound object is added as logo part
                            l.Add(obj);
                            n--;
                            break;
                        }
                    }
                }
                else
                 // is it to considered to be a part of the logo? 
                 if (LogoPartCheck1(obj))
                {
                    l.Add(obj);
                    n--;
                }
                i--;
            }

            // no objects that could be part of the logo.. 
            if (l.Count == 0) return false;

            // remove all candidates that are wrongly positioned
            double w1 = m_MapWidth / 4;
            double w2 = w1 * 2;
            double w3 = w1 * 3;
            double h1 = m_MapHeight / 5;
            double h4 = h1 * 4;
            for (i = 0; i < l.Count; i++)
            {
                MapObject obj = l[i];
                if (obj == null) continue;

                int x = obj.Region.CenterX;
                int y = obj.Region.CenterY;

                if (((x < w2) && (y < h1)) || // upper left 
                    ((x > w3) && (y < h1)) || // upper right
                    ((x < w2) && (y > h4)) || // bottom left
                    ((x > w3) && (y > h4)))   // bottom right
                {
                    // position ok 
                    continue;
                }
                l[i] = null;
            }
            l.RemoveAll(MapObject.IsNull);

            // we now could have multiple groups of objects on different spots
            // the set must be split in those groups then the best fitting is selected as logo.

            // calc avarage width & height
            double w = 0;
            double h = 0;

            foreach (MapObject obj in l)
            {
                w += obj.Region.Width;
                h += obj.Region.Height;
            }
            int iw = (int)((w / l.Count) * 2.0);
            int ih = (int)((h / l.Count) * 2.0);

            List<List<MapObject>> lGroups = new List<List<MapObject>>();

            for (i = 0; i < l.Count; i++)
            {
                MapObject obj = l[i];
                if (l[i] == null) continue;

                List<MapObject> lGroup = new List<MapObject>();
                lGroup.Add(obj);

                Rectangle r = new Rectangle(
                 obj.Region.Left - iw,
                 obj.Region.Top - ih,
                 obj.Region.Width + iw * 2,
                 obj.Region.Height + ih * 2
                );

                for (int j = 0; j < l.Count; j++)
                {
                    if (i == j) continue;
                    MapObject obj2 = l[j];
                    if (obj2 == null) continue;

                    Rectangle r2 = new Rectangle(obj2.Region.Left, obj2.Region.Top, obj2.Region.Width, obj2.Region.Height);
                    if (r.IntersectsWith(r2))
                    {
                        lGroup.Add(obj2);
                        l[j] = null;

                        r = new Rectangle(
                         Math.Min(obj.Region.Left - iw, obj2.Region.Left - iw),
                         Math.Min(obj.Region.Top - ih, obj2.Region.Top - ih),
                         Math.Max(obj.Region.Width + iw * 2, obj2.Region.Width + iw * 2),
                         Math.Max(obj.Region.Height + iw * 2, obj2.Region.Height + ih * 2)
                        );
                    }
                }
                lGroups.Add(lGroup);
            }

            // one group -> nice logo
            if (lGroups.Count == 1)
            {
                foreach (MapObject obj in lGroups[0])
                {
                    if (obj == null) continue;
                    obj.Type = MapObjectType.LogoPart;
                }
            }
            else
             // more groups.. 
             if (lGroups.Count > 1)
            {
                // go for the biggest .... this could be better :=) TODO 
                uint d = 0;
                List<MapObject> lg = null;
                foreach (List<MapObject> g in lGroups)
                {
                    uint t = 0;
                    foreach (MapObject o in g)
                    {
                        if (o != null) t += o.Density;
                    }
                    if (t > d)
                    {
                        d = t;
                        lg = g;
                    }
                }
                if (lg != null)
                {
                    foreach (MapObject o in lg)
                        if (o != null)
                            o.Type = MapObjectType.LogoPart;
                }
            }
            else
              if (lGroups.Count == 0)
            {
                // no group????
                // TODO
                // for know each member in l (all should be remaining) is considered a part of the logo 
                //
                // this is also the case when there is only 1 object, the grouper doesnt work then
                foreach (MapObject obj in l)
                {
                    if (obj == null) continue;
                    obj.Type = MapObjectType.LogoPart;
                }
            }

            return true;
        }

        #endregion
        #endregion

        #region IDisposable Members

        public void Dispose()
        {

        }

        #endregion

        public MapObject MoveDown(IEnumerable<MapObject> targets, MapObject c, int w)
        {
            RECT rc = new RECT(
             c.Region.CenterX - w,
             c.Region.Bottom,
             c.Region.CenterX + w,
             m_MapHeight
            );

            double m = double.MaxValue;
            MapObject o = null;

            foreach (MapObject p in targets)
            {
                if (rc.Contains(p.Region) || rc.OverlapsWith(p.Region))
                {
                    double d = (p.Region.CenterX - c.Region.CenterX) * (p.Region.CenterX - c.Region.CenterX) +
                               (p.Region.CenterY - c.Region.CenterY) * (p.Region.CenterY - c.Region.CenterY);
                    if (d < m)
                    {
                        m = d;
                        o = p;
                    }
                }
                /*    else
                     if (bIncludePartial)
                     {
                      if () l.Add(p);
                     }*/
            }

            return o;
        }

        public MapObject MoveDown(MapObject c, int w)
        {
            return MoveDown(this.Objects, c, w);
        }

        public MapObject MoveDown(MapObject c)
        {
            return MoveDown(this.Objects, c, c.Region.Width);
        }

        public MapObject MoveUp(IEnumerable<MapObject> targets, MapObject c, int w)
        {
            RECT rc = new RECT(
             c.Region.CenterX - w,
             0,
             c.Region.CenterX + w,
             c.Region.Top
            );

            double m = double.MaxValue;
            MapObject o = null;

            foreach (MapObject p in targets)
            {
                if (rc.Contains(p.Region) || rc.OverlapsWith(p.Region))
                {
                    double d = (p.Region.CenterX - c.Region.CenterX) * (p.Region.CenterX - c.Region.CenterX) +
                               (p.Region.CenterY - c.Region.CenterY) * (p.Region.CenterY - c.Region.CenterY);
                    if (d < m)
                    {
                        m = d;
                        o = p;
                    }
                }
            }

            return o;
        }

        public MapObject MoveUp(MapObject c, int w)
        {
            return MoveUp(this.Objects, c, w);
        }

        public MapObject MoveUp(MapObject c)
        {
            return MoveUp(this.Objects, c, c.Region.Width);
        }



        public MapObject MoveRight(IEnumerable<MapObject> targets, MapObject c, int h)
        {
            return MoveRight(targets, h, c.Region.Right, c.Region.CenterY);
        }

        public MapObject MoveRight(IEnumerable<MapObject> targets, int h, int x, int y)
        {
            RECT rc = new RECT(
             x,
             y - h,
             m_MapWidth,
             y + h
            );

            double m = double.MaxValue;
            MapObject o = null;

            foreach (MapObject p in targets)
            {
                if (rc.Contains(p.Region) || rc.OverlapsWith(p.Region))
                {
                    double d = (p.Region.CenterX - x) * (p.Region.CenterX - x) +
                               (p.Region.CenterY - y) * (p.Region.CenterY - y);
                    if (d < m)
                    {
                        m = d;
                        o = p;
                    }
                }
            }

            return o;
        }

        public MapObject MoveRight(IEnumerable<MapObject> targets, MapObject c)
        {
            return MoveRight(targets, c, c.Region.Height);
        }

        public MapObject MoveRight(MapObject c)
        {
            return MoveRight(this.Objects, c, c.Region.Height);
        }

        /// <summary>
        /// find object closest to x,y
        /// </summary>
        public MapObject ClosestTo(int x, int y)
        {
            return ClosestTo(x, y, this.Objects);
        }

        /// <summary>
        /// find object from objects closest to x,y
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="objects"></param>
        /// <returns></returns>
        public MapObject ClosestTo(int x, int y, IEnumerable<MapObject> objects)
        {
            double d = double.MaxValue;
            MapObject o = null;

            if (objects != null)
                foreach (MapObject obj in objects)
                {
                    double dd = (x - obj.Region.CenterX) * (x - obj.Region.CenterX) + (y - obj.Region.CenterY) * (y - obj.Region.CenterY);
                    if (dd < d)
                    {
                        o = obj;
                        d = dd;
                    }
                }

            return o;
        }

        public MapObject ClosestTo(MapObject _object, IEnumerable<MapObject> objects)
        {
            double d = double.MaxValue;
            MapObject o = null;

            double x = _object.Region.CenterX;
            double y = _object.Region.CenterY;

            if (objects != null)
                foreach (MapObject obj in objects)
                {
                    if (_object == obj) continue;

                    double dd = (x - obj.Region.CenterX) * (x - obj.Region.CenterX) + (y - obj.Region.CenterY) * (y - obj.Region.CenterY);
                    if (dd < d)
                    {
                        o = obj;
                        d = dd;
                    }
                }

            return o;
        }


    }
}
