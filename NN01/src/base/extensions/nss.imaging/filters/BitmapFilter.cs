using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS.Imaging
{
 public abstract class BitmapFilter
{
    public abstract bool Apply(Bitmap bmp);
    public abstract bool Supports(PixelFormat format);
}

public abstract class KernelFilter : BitmapFilter
{
    public int DefaultKernelSize { get { return GetDefaultKernelSize(); } }
    public abstract int GetDefaultKernelSize();

    public override bool Apply(Bitmap bmp)
    {
        return Apply(bmp, DefaultKernelSize);
    }
    public abstract bool Apply(Bitmap bmp, int kernel_size);
}

}
