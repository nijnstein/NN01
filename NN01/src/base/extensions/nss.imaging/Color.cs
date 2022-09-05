using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Runtime.InteropServices;

namespace NSS.Imaging
{
    /*
    public abstract class Color
    {
     public abstract byte A { get; }
     public abstract byte R { get; }
     public abstract byte G { get; }
     public abstract byte B { get; }

     public abstract ARGBColor FromArgb(byte a, byte r, byte g, byte b);
     public abstract ARGBColor FromRgb(byte r, byte g, byte b);
    }

    public class ARGBColor : Color
    {
     private byte m_Alpha;
     private byte m_Red;
     private byte m_Green;
     private byte m_Blue;

     public override byte A
     {
      get { return m_Alpha; }
      //   set { m_Alpha = value; }
     }
     public override byte R
     {
      get { return m_Red; }
      //   set { m_Red = value; }
     }
     public override byte G
     {
      get { return m_Green; }
      //   set { m_Green = value; }
     }
     public override byte B
     {
      get { return m_Blue; }
      //   set { m_Blue= value; }
     }
     public ARGBColor(byte a, byte r, byte g, byte b)
     {
      m_Alpha = a;
      m_Red = r;
      m_Green = g;
      m_Blue = b;
     }

     public ARGBColor(short a, short r, short g, short b)
     {
      m_Alpha = (byte)(a >> 8);
      m_Red = (byte)(r >> 8);
      m_Green = (byte)(g >> 8);
      m_Blue = (byte)(b >> 8);
     }

     public override ARGBColor FromArgb(byte a, byte r, byte g, byte b)
     {
      m_Alpha = a;
      m_Red = r;
      m_Green = g;
      m_Blue = b;
      return this;
     }

     public override ARGBColor FromRgb(byte r, byte g, byte b)
     {
      m_Alpha = 255;
      m_Red = r;
      m_Green = g;
      m_Blue = b;
      return this;
     }

    }

    static public class ColorTranslator
    {
     static public Color LabToRGB(int L, int a, int b)
     {
      // For the conversion we first convert values to XYZ and then to RGB
      // Standards used Observer = 2, Illuminant = D65

      const double ref_X = 95.047;
      const double ref_Y = 100.000;
      const double ref_Z = 108.883;

      double var_Y = ((double)L + 16.0) / 116.0;
      double var_X = (double)a / 500.0 + var_Y;
      double var_Z = var_Y - (double)b / 200.0;

      if (Math.Pow(var_Y, 3) > 0.008856)
       var_Y = Math.Pow(var_Y, 3);
      else
       var_Y = (var_Y - 16 / 116) / 7.787;

      if (Math.Pow(var_X, 3) > 0.008856)
       var_X = Math.Pow(var_X, 3);
      else
       var_X = (var_X - 16 / 116) / 7.787;

      if (Math.Pow(var_Z, 3) > 0.008856)
       var_Z = Math.Pow(var_Z, 3);
      else
       var_Z = (var_Z - 16 / 116) / 7.787;

      double X = ref_X * var_X;
      double Y = ref_Y * va
      r_Y;
      double Z = ref_Z * var_Z;

      return XYZToRGB(X, Y, Z);
     }

     static public Color XYZToRGB(double X, double Y, double Z)
     {
      // Standards used Observer = 2, Illuminant = D65
      // ref_X = 95.047, ref_Y = 100.000, ref_Z = 108.883

      double var_X = X / 100.0;
      double var_Y = Y / 100.0;
      double var_Z = Z / 100.0;

      double var_R = var_X * 3.2406 + var_Y * (-1.5372) + var_Z * (-0.4986);
      double var_G = var_X * (-0.9689) + var_Y * 1.8758 + var_Z * 0.0415;
      double var_B = var_X * 0.0557 + var_Y * (-0.2040) + var_Z * 1.0570;

      if (var_R > 0.0031308)
       var_R = 1.055 * (Math.Pow(var_R, 1 / 2.4)) - 0.055;
      else
       var_R = 12.92 * var_R;

      if (var_G > 0.0031308)
       var_G = 1.055 * (Math.Pow(var_G, 1 / 2.4)) - 0.055;
      else
       var_G = 12.92 * var_G;

      if (var_B > 0.0031308)
       var_B = 1.055 * (Math.Pow(var_B, 1 / 2.4)) - 0.055;
      else
       var_B = 12.92 * var_B;

      int nRed = (int)(var_R * 256.0);
      int nGreen = (int)(var_G * 256.0);
      int nBlue = (int)(var_B * 256.0);

      if (nRed < 0) nRed = 0;
      else if (nRed > 255) nRed = 255;
      if (nGreen < 0) nGreen = 0;
      else if (nGreen > 255) nGreen = 255;
      if (nBlue < 0) nBlue = 0;
      else if (nBlue > 255) nBlue = 255;

      return new ARGBColor((byte)255, (byte)nRed, (byte)nGreen, (byte)nBlue); ;
     }

     static public Color CMYKToRGB(double C, double M, double Y, double K)
     {
      int nRed = (int)((1.0 - (C * (1 - K) + K)) * 255);
      int nGreen = (int)((1.0 - (M * (1 - K) + K)) * 255);
      int nBlue = (int)((1.0 - (Y * (1 - K) + K)) * 255);

      if (nRed < 0) nRed = 0;
      else if (nRed > 255) nRed = 255;
      if (nGreen < 0) nGreen = 0;
      else if (nGreen > 255) nGreen = 255;
      if (nBlue < 0) nBlue = 0;
      else if (nBlue > 255) nBlue = 255;

      return new ARGBColor((byte)255, (byte)nRed, (byte)nGreen, (byte)nBlue); ;
     }
    }
     */

    [StructLayout(LayoutKind.Explicit)]
    public struct Color24
    {
        [FieldOffset(0)]
        public byte B;
        [FieldOffset(1)]
        public byte G;
        [FieldOffset(2)]
        public byte R;

        public Color Color
        {
            get { return Color.FromArgb(255, R, G, B); }
        }
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct Color32
    {
        [FieldOffset(0)]
        public byte B;
        [FieldOffset(1)]
        public byte G;
        [FieldOffset(2)]
        public byte R;
        [FieldOffset(3)]
        public byte A;
        [FieldOffset(0)]
        public uint ARGB;
        public Color Color
        {
            get { return Color.FromArgb(A, R, G, B); }
        }
    }

    static public class IndexedColor
    {
        #region Extract White Values from Palette

        static public uint GetWhiteValue_1BPP(Bitmap bmp)
        {
            uint white_value = 0xFFFFFFFF;

            int i1 = bmp.Palette.Entries[0].R + bmp.Palette.Entries[0].G + bmp.Palette.Entries[0].B;
            int i2 = bmp.Palette.Entries[1].R + bmp.Palette.Entries[1].G + bmp.Palette.Entries[1].B;
            if (i1 > i2) white_value = 0x00000000;

            return white_value;
        }

        static public List<Byte> GetWhiteValue_8BPP(Bitmap bmp, ref byte min_white, ref byte max_white, int threshold)
        {
            List<byte> white_list = new List<byte>();

            min_white = 255;
            max_white = 0;

            for (int i = 0; i < 256; i++)
            {
                System.Drawing.Color c = bmp.Palette.Entries[i];

                if ((c.R > threshold) &&
                    (c.G > threshold) &&
                    (c.B > threshold))
                {
                    white_list.Add((byte)i);

                    min_white = (byte)Math.Min(min_white, i);
                    max_white = (byte)Math.Max(max_white, i);
                }
            }

            return white_list;
        }
        #endregion

    }
}
