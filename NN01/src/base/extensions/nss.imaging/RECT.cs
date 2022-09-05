

using System.Drawing;

namespace NSS.Imaging
{
    public struct RECT
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;

        public int Width
        {
            get
            {
                if (Right > Left) return Right - Left; else return Left - Right;
            }
        }

        public int Height
        {
            get
            {
                if (Bottom > Top) return Bottom - Top; else return Top - Bottom;
            }
        }

        static public bool IsNullOrEmpty(RECT rc)
        {
            return RECT.IsInvalid(rc) || RECT.IsNull(rc) || (rc.Surface == 0);
        }

        static public RECT Null
        {
            get
            {
                return new RECT(0, 0, 0, 0);
            }
        }
        static public bool IsNull(RECT rc)
        {
            // bottom right first, less likely to be 0 ;)
            return (rc.Right == 0) && (rc.Bottom == 0) && (rc.Left == 0) && (rc.Top == 0);
        }

        static public RECT Invalid
        {
            get
            {
                return new RECT(int.MinValue, int.MinValue, int.MinValue, int.MinValue);
            }
        }
        static public bool IsInvalid(RECT rc)
        {
            return (rc.Right == int.MinValue) && (rc.Bottom == int.MinValue) && (rc.Left == int.MinValue) && (rc.Top == int.MinValue);
        }

        public RECT(int x, int y, int x2, int y2)
        {
            Left = x;
            Right = x2;
            Top = y;
            Bottom = y2;
        }

        public static RECT FromLRTB(int left, int right, int top, int bottom) => new RECT(left, top, right, bottom);

        public int Surface
        {
            get
            {
                int w;
                if (Left < Right) w = Right - Left;
                else w = Left - Right;

                int h;
                if (Top < Bottom) h = Bottom - Top;
                else h = Top - Bottom;

                return w * h;
            }
        }

        public int CenterX
        {
            get
            {
                return (int)((Left + Right) / 2);
            }
        }

        public int CenterY
        {
            get
            {
                return (int)((Top + Bottom) / 2);
            }
        }

        public int BoundingRadius
        {
            get
            {
                float w = Width;
                float h = Height;
                float d = MathF.Sqrt((w * w) + (h * h));
                return (int)(d / 2);
            }
        }

        public Rectangle Rectangle
        {
            get
            {
                return new Rectangle
                (
                 Math.Min(Left, Right),
                 Math.Min(Top, Bottom),
                 Width,
                 Height
                );
            }
        }

        public RectangleF RectangleF
        {
            get
            {
                return new Rectangle
                (
                 Math.Min(Left, Right),
                 Math.Min(Top, Bottom),
                 Width,
                 Height
                );
            }
        }

       

        public bool OverlapsWith(RECT rc)
        {
            return RectangleF.IntersectsWith(rc.RectangleF);
        }

        /// <summary>
        /// Returns if the given rectangle is completely contained within this rectangle 
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="y1"></param>
        /// <param name="x2"></param>
        /// <param name="y2"></param>
        /// <returns></returns>
        public bool Contains(int x1, int y1, int x2, int y2)
        {
            return (x1 >= Left) && (x2 <= Right) && (y1 >= Top) && (y2 <= Bottom) &&
                   (x2 > Left) && (x1 < Right) && (y2 > Top) && (y1 < Bottom);
        }

        /// <summary>
        /// Returns if the given rectangle is completely contained within this rectangle 
        /// </summary>
        /// <param name="rc"></param>
        /// <returns></returns>
        public bool Contains(RECT rc)
        {
            return (rc.Left >= Left) && (rc.Right <= Right) && (rc.Top >= Top) && (rc.Bottom <= Bottom) &&
                   (rc.Right > Left) && (rc.Left < Right) && (rc.Bottom > Top) && (rc.Top < Bottom);
        }
    }
}
