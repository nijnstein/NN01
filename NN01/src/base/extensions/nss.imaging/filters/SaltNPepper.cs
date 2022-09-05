using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;

namespace NSS.Imaging
{
    public class SaltNPepper : BitmapFilter
 {
  /// <summary>
  /// Salt n pepper filter, removes single pixels & fills single pixel holes :) 
  /// ONLY USED FOR 1-BIT IMAGE DATA
  /// </summary>
  public override bool Apply(Bitmap bmp)
  {
   if (bmp == null) return false;
   if (bmp.PixelFormat != PixelFormat.Format1bppIndexed) return false;

   Rectangle rcLock = new Rectangle(0, 0, bmp.Width - 1, bmp.Height - 1);
   BitmapData bmd = bmp.LockBits(rcLock, ImageLockMode.ReadWrite, PixelFormat.Format1bppIndexed);
   try
   {
    unsafe
    {
     return SaltNPepper_1BPP((uint*)bmd.Scan0.ToPointer(), bmd.Stride, bmp.Width, bmp.Height);
    }
   }
   finally
   {
    bmp.UnlockBits(bmd);
   }   
  }

  public override bool Supports(PixelFormat format)
  {
   return format == PixelFormat.Format1bppIndexed; 
  }


#if WIN32

        /// <summary>
        /// Unsafe Native Memory Copy
        /// </summary>
        /// <param name="dest">destination address</param>
        /// <param name="src">source address</param>
        /// <param name="length">length in bytes</param>
        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
  unsafe public static extern void CopyMemory(void* dest, void* src, int length);

#endif


        // mm todo  there is a bug somewhere here..
        unsafe static public bool SaltNPepper_1BPP(uint* scan0, int stride, int width, int height)
  {
   if (scan0 == null) return false;
   if (width <= 3 || height <= 3) return false;

   int w = (width + 31) >> 5;
   uint b_in0, b_in1, b_in2, b_in0_prev, b_in1_prev, b_in2_prev, b_in0_next, b_in1_next, b_in2_next;
   uint p0, p1, p2, p3, p4, p5, p6, p7, p8, c;

   uint* back_write = stackalloc uint[w];
   uint* back_write_prev = stackalloc uint[w];

   for (int y = 1; y < height - 2; y++)
   {
    // get pointers to starts of scanlines at x=1
    uint* p_in0 = scan0 + (y - 1) * stride / 4;
    uint* p_in1 = p_in0 + stride / 4;
    uint* p_in2 = p_in0 + stride / 4 + stride / 4;

    // the pointer to the backwrite buffer, initialize with white pixels to start with
    // this has as effect that the border margin is blanked.. TODO  care??
    *back_write = ~(uint)0;
    uint* p_out = back_write + 1;

    //pre initialize pixel blocks for the reuse 
    b_in0 = (uint)~*(p_in0);
    b_in0_next = (uint)~*(p_in0 + 1);

    b_in1 = (uint)~*(p_in1);
    b_in1_next = (uint)~*(p_in1 + 1);

    b_in2 = (uint)~*(p_in2);
    b_in2_next = (uint)~*(p_in2 + 1);

    // advance 1 word to match loop start (x)
    p_in0++;
    p_in1++;
    p_in2++;

    for (int x = 1; x < (w - 1); x++)
    {
     // get new pixel blocks, reuse the others
     b_in0_prev = b_in0;
     b_in0 = b_in0_next;
     b_in0_next = (uint)~*(p_in0 + 1);

     b_in1_prev = b_in1;
     b_in1 = b_in1_next;
     b_in1_next = (uint)~*(p_in1 + 1);

     b_in2_prev = b_in2;
     b_in2 = b_in2_next;
     b_in2_next = (uint)~*(p_in2 + 1);

     // shuffle the new pixel blocks
     b_in0_next = ((b_in0_next & 0x000000FF) << 24) | ((b_in0_next & 0x0000FF00) << 8) | ((b_in0_next & 0x00FF0000) >> 8) | ((b_in0_next & 0xFF000000) >> 24);
     b_in1_next = ((b_in1_next & 0x000000FF) << 24) | ((b_in1_next & 0x0000FF00) << 8) | ((b_in1_next & 0x00FF0000) >> 8) | ((b_in1_next & 0xFF000000) >> 24);
     b_in2_next = ((b_in2_next & 0x000000FF) << 24) | ((b_in2_next & 0x0000FF00) << 8) | ((b_in2_next & 0x00FF0000) >> 8) | ((b_in2_next & 0xFF000000) >> 24);

     // pre init pixel values.. 
     if (x == 0)
     {
      p1 = 0;
      p2 = 0;
      p4 = 0;
      p5 = 0;
      p7 = 0;
      p8 = 0;
     }
     else
     {
      p1 = (uint)(b_in0 & 1);
      p4 = (uint)(b_in1 & 1);
      p7 = (uint)(b_in2 & 1);
      p2 = (uint)(b_in0 & 2) >> 1;
      p5 = (uint)(b_in1 & 2) >> 1;
      p8 = (uint)(b_in2 & 2) >> 1;
     }

     uint b_out = 0;
     for (int i = 0; i < 32; i++)
     {
      // cycle pixelvalues..
      p0 = p1;
      p1 = p2;
      p3 = p4;
      p4 = p5;
      p6 = p7;
      p7 = p8;

      if (i < 31)
      {
       int shift = i + 1;
       int shifted = 1 << shift;
       p2 = (uint)(b_in0 & shifted) >> shift;
       p5 = (uint)(b_in1 & shifted) >> shift;
       p8 = (uint)(b_in2 & shifted) >> shift;
      }
      else
      {
       if (x < w)
       {
        p2 = (uint)(b_in0_next & 1);
        p5 = (uint)(b_in1_next & 1);
        p8 = (uint)(b_in2_next & 1);
       }
       else
       {
        p2 = 0;
        p5 = 0;
        p8 = 0;
       }
      }

      c = p0 + p1 + p2 + p3 + p5 + p6 + p7 + p8;

      /// AMOUNT OF CONNECTEDNESS THRESSHOLD.. 
      /// originally 0 & 8, 1 and 7 give better results though TODO -> test more
      /// problem with 1 and 7 is that it touches endpoints of small features... 
      /// however that type of distortion will partly be fixed in the last stage
      /// where mapdata is used together with morphological operators for
      /// reconstruction/recombination of small features (characters)
      if (c <= 1)
      {
       b_out = b_out & (uint)~(1 << i);
      }
      else
       if (c >= 7)
       {
        b_out = b_out | (uint)(1 << i);
       }
       else
       {
        if (p4 > 0)
        {
         b_out = b_out | (uint)(1 << i);
        }
        else
        {
         b_out = b_out & (uint)~(1 << i);
        }
       }
     }

     b_out = ((b_out & 0x000000FF) << 24) | ((b_out & 0x0000FF00) << 8) | ((b_out & 0x00FF0000) >> 8) | ((b_out & 0xFF000000) >> 24);
     *p_out = (uint)~b_out;

     p_out++;
     p_in0++;
     p_in1++;
     p_in2++;
    }

    // write back pixel block
    if (y > 1)
    {
     uint* pout = scan0 + y * stride / 4;
     uint* pin = back_write_prev;
     for (int ip = 0; ip < w; ip++) *pout++ = *pin++;
    }
    back_write_prev = back_write;
   }

   // write back last pixel block
   {
    uint* pout = scan0 + (height - 1) * stride / 4;
    uint* pin = back_write_prev;

    for (int ip = 0; ip < w; ip++) *pout++ = *pin++;
   }
   return true;
  }

 }
}
