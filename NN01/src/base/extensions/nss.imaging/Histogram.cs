using NSS;
using System.Drawing;

namespace NSS
{
    public class Histogram
    {
        int binCount;
        int[] bins;
        List<float> samples;

        float stepSize;
        float mean;
        float sd;
        int nsd;

        float min;
        float max;

        /// <summary>
        /// plot a histogram from samples binned on SD 
        /// </summary>
        /// <param name="mean">mean of distribution to measure</param>
        /// <param name="sd">1 standard deviation</param>
        /// <param name="Nsd">the number of sd's minus and positive binned into a histogram</param>
        /// <param name="stepsPerSD">number of steps to bin for each Sd range</param>
        public Histogram(float mean, float sd, int Nsd, int stepsPerSD)
        {
            this.mean = mean;
            this.sd = sd;
            this.nsd = Nsd * stepsPerSD;

            binCount = stepsPerSD * Nsd * 2 + 1;
            bins = new int[binCount];
            bins.Zero();
            stepSize = (sd / stepsPerSD);

            min = -Nsd * sd + mean;
            max = Nsd * sd + mean;

            samples = new List<float>();
        }

        public int AddSample(float f)
        {
            samples.Add(f);

            if (f < min)
            {
                bins[0]++;
                return 0;
            }
            else
            if (f >= max)
            {
                bins[binCount - 1]++;
                return binCount - 1;
            }
            else
            {
                int ibin = ((f - mean) / stepSize).FloorToInt() + nsd;
                bins[ibin]++;
                return ibin;
            }
        }

        public float Average => samples.Average();
        public float Variance => MathEx.Variance(samples.ToArray(), Average);
        public float SD => MathF.Sqrt(MathEx.Variance(samples.ToArray(), Average));
        public int SampleCount => MathEx.Sum(bins);

        public Bitmap GenerateBitmap(int width, int height, string? title = null)
        {
            Bitmap bmp = new Bitmap(width, height);

            Graphics g = Graphics.FromImage(bmp);
            g.FillRectangle(Brushes.White, 0, 0, width, height);

            // margin from image border of axes 
            int margin_x = 5;
            int margin_y = 20;

            int x_axes_y = height - margin_y * 4;
            int x_axes_x1 = margin_x;
            int x_axes_x2 = width - margin_x;

            int y_axes_x = width / 2;
            int y_axes_y1 = margin_y * 4;
            int y_axes_y2 = height - margin_y * 4;

            Pen axesPen = new Pen(Color.Black, 1.5f * (height / 400));
            Font font = new Font("Calibri", 10 * (height / 400));

            // draw axes 
            g.DrawLine(axesPen, x_axes_x1, x_axes_y, x_axes_x2, x_axes_y);
            g.DrawLine(axesPen, y_axes_x, y_axes_y1, y_axes_x, y_axes_y1);

            int x_space_per_bin = (x_axes_x2 - x_axes_x1) / binCount;
            float max_y_per_bin = y_axes_y2 - y_axes_y1;
            float max = bins.Max();

            // draw bins 
            for (int i = 0; i < binCount; i++)
            {
                float y = (max_y_per_bin / max) * bins[i];
                Color color = Color.FromArgb((int)Random.Shared.Range(0, 255), (int)Random.Shared.Range(0, 255), (int)Random.Shared.Range(0, 255));

                g.FillRectangle(
                    new SolidBrush(color),
                    x_axes_x1 + i * x_space_per_bin,
                    y_axes_y2 - y,
                    x_space_per_bin,
                    y);
            }

            // draw SD lines 
            int b_min_sd = ((-sd) / stepSize).FloorToInt() + nsd;
            int b_plus_sd = ((sd) / stepSize).FloorToInt() + nsd;

            int x = b_min_sd * x_space_per_bin + x_axes_x1;
            g.DrawLine(axesPen, x, x_axes_y, x, margin_y * 4);

            SizeF sz = g.MeasureString($"-SD({-sd})", font);
            g.DrawString($"-SD({-sd})", font, Brushes.Black, x - (sz.Width / 2f), x_axes_y + margin_y - sz.Height / 2f);

            x = b_plus_sd * x_space_per_bin + x_axes_x1;
            g.DrawLine(axesPen, x, x_axes_y, x, margin_y * 4);

            sz = g.MeasureString($"+SD({sd})", font);
            g.DrawString($"+SD({sd})", font, Brushes.Black, x - (sz.Width / 2f), x_axes_y + margin_y - sz.Height / 2f);

            x = nsd * x_space_per_bin + x_axes_x1;
            g.DrawLine(axesPen, x, x_axes_y, x, margin_y * 4);

            sz = g.MeasureString($"MEAN({mean})", font);
            g.DrawString($"MEAN({mean})", font, Brushes.Black, x - (sz.Width / 2f), x_axes_y + margin_y - sz.Height / 2f);


            // any title 
            sz = g.MeasureString(title ?? "histogram", font);
            g.DrawString(title ?? "histogram", font, Brushes.Black, width / 2 - sz.Width / 2, 2);

            string info = $"avg = {Average.ToString("0.000")}, var = {Variance.ToString("0.000")}, sd = {SD.ToString("0.000")}, n = {SampleCount}";
            sz = g.MeasureString(info, font);
            g.DrawString(info, font, Brushes.Black, width / 2 - sz.Width / 2, sz.Height + 4);

            return bmp;
        }
    }
}
                                            