using NSS;
using NSS.Neural;
using System.Drawing;

namespace NN01 
{
    static public class DigitSampleSets
    {
        public static int LoadDigitGrid(
            this SampleSet set, 
            int fromIndex, // index to start filling the set from 
            string filename, // image filename
            int klass,
            int w = 28, 
            int h = 28,
            int c = 980)
        {                           
            int s = w * h;
            int sampleIndex = fromIndex;
            int i = 0; 

            using (Bitmap bmp = (Bitmap)Bitmap.FromFile(filename))
            {
                for (int x = 0; x < bmp.Width / w && i < c; x++)
                {
                    for(int y = 0; y < bmp.Height / h && i < c; y++)
                    {
                        // copy image data into 1d vector, leading x 
                        Span<float> sampleData = set.Data.AsSpan2D<float>().Row(sampleIndex);

                        int j = 0; 
                        for (int iy = y * h; iy < y * h + h; iy++)
                        {
                            for (int ix = x * w; ix < x * w + w; ix++)
                            {
                                sampleData[j++] = bmp.GetPixel(ix, iy).GetBrightness(); // mm performance will be great....
                            }
                        }

                        // only train if the region has some filling
                        if (MathEx.Max(sampleData) > 0.2f)
                        {
                            set.Samples[sampleIndex] = new Sample(sampleIndex, set.SampleSize, klass, set.ClassCount);
                            sampleIndex++;
                            i++;
                        }
                    }
                }
            }
            return sampleIndex;
        }

    }
}
