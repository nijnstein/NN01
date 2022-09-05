using NSS;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NSS.Imaging
{
 static public class DistanceTransform
{

    static public double[] Run(double[] sample, int w, int h, double set, double notset)
    {
        double[] r = new double[sample.Length];
        double[] c = new double[sample.Length];


        // row scan 
        for (int y = 0; y < h; y++)
        {
            int last = 0;
            int next = 0;
            for (int x = 0; x < w; x++)
            {
                double d = sample[y * w + x];
                if (d == set)
                {
                    if (next <= x)
                        for (int x2 = x + 1; x2 < w; x2++)
                            if (sample[y * w + x2] == notset)
                            {
                                next = x2;
                                break;
                            }

                    int min = (last > 0) ? (x - last) : 0;
                    if (next > 0) min = Math.Min(min, next - x) + 1;

                    r[y * w + x] = min * min;
                }
                else
                {
                    r[y * w + x] = 0;
                    last = x;
                }
            }
        }

        // column scan 
        for (int x = 0; x < w; x++)
        {
            int last = 0;
            int next = 0;
            for (int y = 0; y < h; y++)
            {
                double d = sample[y * w + x];
                if (d == set)
                {
                    if (next <= y)
                        for (int y2 = y + 1; y2 < h; y2++)
                            if (sample[y2 * w + x] == notset)
                            {
                                next = y2;
                                break;
                            }

                    int min = (last > 0) ? (x - last) : 0;
                    if (next > 0) min = Math.Min(min, next - x) + 1;

                    c[y * w + x] = min * min;
                }
                else
                {
                    c[y * w + x] = 0;
                    last = y;
                }
            }
        }

        // todo.. diagonals..
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                r[y * w + x] = Math.Min(r[y * w + x], c[y * w + x]);

        //
        MathEx.NormalizeLog10(r, 0, 1);
        return r;
    }


    static public double[] Run(double[] sample, int w, int h, double set_low, double set_high, double notset_low, double notset_high)
    {
        double[] r = new double[sample.Length];
        double[] c = new double[sample.Length];

        // row scan 
        for (int y = 0; y < h; y++)
        {
            int last = 0;
            int next = 0;
            for (int x = 0; x < w; x++)
            {
                double d = sample[y * w + x];
                if (d >= set_low && d <= set_high)
                {
                    if (next <= x)
                        for (int x2 = x + 1; x2 < w; x2++)
                        {
                            d = sample[y * w + x2];
                            if (d >= notset_low && d <= notset_high)
                            {
                                next = x2;
                                break;
                            }
                        }
                    int min = (last > 0) ? (x - last) : 0;
                    if (next > 0) min = Math.Min(min, next - x) + 1;

                    r[y * w + x] = min * min;
                }
                else
                {
                    r[y * w + x] = 0;
                    last = x;
                }
            }
        }

        // column scan 
        for (int x = 0; x < w; x++)
        {
            int last = 0;
            int next = 0;
            for (int y = 0; y < h; y++)
            {
                double d = sample[y * w + x];
                if (d >= set_low && d <= set_high)
                {
                    if (next <= y)
                        for (int y2 = y + 1; y2 < h; y2++)
                        {
                            d = sample[y2 * w + x];
                            if (d >= notset_low && d <= notset_high)
                            {
                                next = y2;
                                break;
                            }
                        }

                    int min = (last > 0) ? (x - last) : 0;
                    if (next > 0) min = Math.Min(min, next - x) + 1;

                    c[y * w + x] = min * min;
                }
                else
                {
                    c[y * w + x] = 0;
                    last = y;
                }
            }
        }

        // todo.. diagonals..
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                r[y * w + x] = Math.Min(r[y * w + x], c[y * w + x]);

        //
        MathEx.NormalizeLog10(r, notset_low, set_high);
        return r;
    }



}
}
