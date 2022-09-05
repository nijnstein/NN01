
using System.Drawing;

namespace NSS.Imaging.Windows 
{
    public static class MakeTransparentGIF
{

        /// <summary>
        /// make a gif transparent by altering its color table directly, .net components dont work for gif
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="color">color to make transparent</param>
        /// <returns></returns>
        static public Bitmap MakeTransparentGif(this Bitmap bitmap, Color color)
        {
            byte R = color.R;
            byte G = color.G;
            byte B = color.B;

            MemoryStream ms_in = new MemoryStream();
            bitmap.Save(ms_in, System.Drawing.Imaging.ImageFormat.Gif);

            MemoryStream ms_out = new MemoryStream((int)ms_in.Length);
            int count = 0;
            byte[] buf = new byte[256];
            byte transparentIdx = 0;
            ms_in.Seek(0, SeekOrigin.Begin);

            // check the header
            count = ms_in.Read(buf, 0, 13);
            if ((buf[0] != 71) || (buf[1] != 73) || (buf[2] != 70)) return null; //GIF
            ms_out.Write(buf, 0, 13);
            int i = 0;
            if ((buf[10] & 0x80) > 0)
            {
                i = 1 >> ((buf[10] & 7) + 1) == 256 ? 256 : 0;
            }
            for (; i != 0; i--)
            {
                ms_in.Read(buf, 0, 3);
                if ((buf[0] == R) && (buf[1] == G) && (buf[2] == B))
                {
                    transparentIdx = (byte)(256 - i);
                }
                ms_out.Write(buf, 0, 3);
            }
            bool gcePresent = false;
            while (true)
            {
                ms_in.Read(buf, 0, 1);
                ms_out.Write(buf, 0, 1);
                if (buf[0] != 0x21) break;
                ms_in.Read(buf, 0, 1);
                ms_out.Write(buf, 0, 1);
                gcePresent = (buf[0] == 0xf9);
                while (true)
                {
                    ms_in.Read(buf, 0, 1);
                    ms_out.Write(buf, 0, 1);
                    if (buf[0] == 0) break;
                    count = buf[0];
                    if (ms_in.Read(buf, 0, count) != count) return null;
                    if (gcePresent)
                    {
                        if (count == 4)
                        {
                            buf[0] |= 0x01;
                            buf[3] = transparentIdx;
                        }
                    }
                    ms_out.Write(buf, 0, count);
                }
            }
            while (count > 0)
            {
                count = ms_in.Read(buf, 0, 1);
                ms_out.Write(buf, 0, 1);
            }
            ms_in.Close();
            ms_out.Flush();
            return new Bitmap(ms_out);
        }

    }
}
