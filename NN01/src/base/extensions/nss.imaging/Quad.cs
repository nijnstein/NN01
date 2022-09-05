using NSS.Imaging;
 
namespace NSS 
{
    public struct QUAD
    {
        public int x1, y1, x2, y2, x3, y3, x4, y4;

        public QUAD(int _x1, int _y1, int _x2, int _y2, int _x3, int _y3, int _x4, int _y4)
        {
            x1 = _x1;
            x2 = _x2;
            x3 = _x3;
            x4 = _x4;
            y1 = _y1;
            y2 = _y2;
            y3 = _y3;
            y4 = _y4;
        }

        public QUAD(RECT rc)
        {
            x1 = rc.Left;
            x2 = rc.Right;
            x3 = rc.Left;
            x4 = rc.Right;
            y1 = rc.Top;
            y2 = rc.Top;
            y3 = rc.Bottom;
            y4 = rc.Bottom;
        }

        public RECT OuterRectangle
        {
            get
            {
                return  RECT.FromLRTB(
                 MathEx.Min(x1, x2, x3, x4),
                 MathEx.Min(y1, y2, y3, y4),
                 MathEx.Max(x1, x2, x3, x4),
                 MathEx.Max(y1, y2, y3, y4));
            }
        }

        public int Width
        {
            get
            {
                return MathEx.Max(x1, x2, x3, x4) - MathEx.Min(x1, x2, x3, x4);
            }
        }

        public int Height
        {
            get
            {
                return MathEx.Max(y1, y2, y3, y4) - MathEx.Min(y1, y2, y3, y4);
            }
        }

        public int Surface
        {
            get
            {
                return
                 x1 * y2 - y1 * x2 +
                 x2 * y4 - y2 * x4 +
                 x4 * y3 - y4 * x3 +
                 x3 * y1 - y3 * x1;
            }
        }


    }
}
