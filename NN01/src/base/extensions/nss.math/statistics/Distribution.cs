using NSS;

namespace NSS
{
    public class Distribution
    {
        protected List<uint> Samples = new List<uint>();
        public int SampleCount
        {
            get
            {
                if (Samples == null) return 0;
                return Samples.Count;
            }
        }

        public uint Max = 0;
        public uint Min = uint.MaxValue;
        public uint Total = 0;

        public uint Average = 0;

        public uint MedianMin = uint.MaxValue;
        public uint MedianMax = 0;
        public uint MedianTotal = 0;
        public uint MedianCount = 0;
        public uint MedianAverage = 0;

        public double SD = 0;

        public void Clear()
        {
            Max = 0;
            Min = uint.MaxValue;
            Total = 0;
            Average = 0;

            MedianTotal = 0;
            MedianCount = 0;
            MedianMin = uint.MaxValue;
            MedianMax = 0;
            MedianAverage = 0;

            SD = 0;

            Samples.Clear();
        }

        public void ClearSamples()
        {
            Samples.Clear();
        }

        public void AddSample(uint sample)
        {
            Samples.Add(sample);

            Total += sample;
            Max = Math.Max(sample, Max);
            Min = Math.Min(sample, Min);
        }

        public void Calculate()
        {
            Calculate(1, 1);
        }
        public void Calculate(double min_border_factor, double max_border_factor)
        {
            uint c = (uint)SampleCount;
            if (c == 0) return;

            Average = Total / c;

            uint min = (uint)(min_border_factor * Min);
            uint max = (uint)(max_border_factor * Max);

            MedianMin = uint.MaxValue;
            MedianMax = 0;
            MedianCount = 0;
            MedianAverage = 0;
            MedianTotal = 0;

            double d = 0.0;

            foreach (uint sample in Samples)
            {
                if (sample < min) continue;
                if (sample > max) continue;

                MedianTotal += sample;
                MedianCount++;

                MedianMax = Math.Max(sample, MedianMax);
                MedianMin = Math.Min(sample, MedianMin);

                d += sample * sample;
            }

            if (MedianCount > 0)
            {
                MedianAverage = MedianTotal / MedianCount;

                d = d / MedianCount;
                SD = Math.Sqrt(d - MedianAverage * MedianAverage);
            }
            else
            {
                MedianAverage = 0;
                SD = 0;
            }
        }

    }
}
