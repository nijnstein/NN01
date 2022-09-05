using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS.Imaging
{
    public static class RectangleExtensions
    {
        static public RECT ToRECT(this System.Drawing.Rectangle rc)
        {
            return new RECT(rc.Left, rc.Top, rc.Left + rc.Width, rc.Top + rc.Bottom);
        }

        static public bool Inside(this System.Drawing.Rectangle rc, System.Drawing.Rectangle rc2)
        {
            return rc.Left > rc2.Left && rc.Top > rc2.Top && rc.Right < rc2.Left && rc.Bottom < rc2.Bottom;
        }

        static public System.Drawing.Rectangle Scale(this System.Drawing.Rectangle rc, double scale)
        {
            return new System.Drawing.Rectangle(
             (int)(rc.Left * scale),
             (int)(rc.Top * scale),
             (int)(rc.Width * scale),
             (int)(rc.Height * scale));
        }

        static public bool IsNullOrEmpty(this System.Drawing.Rectangle rc)
        {
            return rc == System.Drawing.Rectangle.Empty;
        }
    }
}