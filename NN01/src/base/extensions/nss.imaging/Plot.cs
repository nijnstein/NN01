using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{

    public class Plot
    {
        public int Width;
        public int Height;

        public struct GraphPoint
        {
            public float X;
            public float Y;
            public int K;
            public GraphPoint(float x, float y, int k)
            {
                X = x;
                Y = y;
                K = k;
            }
        }

        List<GraphPoint> Points = new List<GraphPoint>();

        public Plot()
        {
        }

        public void AddPoint(float x, float y, int k)
        {
            Points.Add(new GraphPoint(x, y, k));
        }

        public void Render(Bitmap bmp, int centerx, int centery, int w, int h, int scalex, int scaley)
        {
            if (bmp == null) return;
            if (w <= 0 || h <= 0 || scalex == 0 || scaley == 0) return;
            if (centerx + w < bmp.Width || centery + h < bmp.Height) return;

            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.FillRectangle(SystemBrushes.Window, 0, 0, w, h);
                g.DrawLine(Pens.Black, 0, centery, w, centery);
                g.DrawLine(Pens.Black, centerx, 0, centerx, h);
                foreach (GraphPoint p in Points)
                {
                    float x = centerx + scalex * p.X;
                    float y = centery + scaley * -p.Y;
                    if (x >= bmp.Width - 1) continue;
                    if (y > bmp.Height - 1) continue;

                    g.DrawRectangle(Pens.Red, (int)x, (int)y, 1, 1);
                }
            }
        }

        public void RenderLines(Bitmap bmp, int centerx, int centery, int w, int h, int scalex, int scaley)
        {
            if (bmp == null) return;
            if (w <= 0 || h <= 0 || scalex == 0 || scaley == 0) return;
            if (centerx + w < bmp.Width || centery + h < bmp.Height) return;

            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.FillRectangle(SystemBrushes.Window, 0, 0, w, h);
                g.DrawLine(Pens.Black, 0, centery, w, centery);
                g.DrawLine(Pens.Black, centerx, 0, centerx, h);
                foreach (GraphPoint p in Points)
                {
                    float x = centerx + scalex * (float)p.X;
                    float y = centery + scaley * (float)-p.Y;
                    float y1 = centery;
                    if (x >= bmp.Width - 1) continue;
                    if (y > bmp.Height - 1) continue;

                    g.DrawLine(Pens.Red, (int)x, (int)y1, (int)x, (int)y);
                }
            }
        }
    }
}