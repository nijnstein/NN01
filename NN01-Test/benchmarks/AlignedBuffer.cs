using BenchmarkDotNet.Attributes;
using NN01;
using NSS;
using NUnit.Framework;
using System.Buffers;

namespace Test
{

    public class AlignedBufferBenchmark
    {
        static public float[] a = new float[]
        {
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001
        };

        private AlignedBuffer<float> buffer = new AlignedBuffer<float>(a.Length, 32, true);
        private float[] fbuffer = new float[1000];

        [GlobalSetup]
        public void InitBuffer()
        {
            buffer.With((lease) =>
            {
                Span<float> s = lease.AsSpan();

                for (int i = 0; i < s.Length; i++)
                {
                    s[i] = 2f;
                }
            });
        }

        [Benchmark]
        public void SumAvxAlignedx1000() =>
            buffer.With((lease) =>
            {
                for (int i = 0; i < 1000; i++)
                {
                    Intrinsics.Sum(lease.GetSpan(0, a.Length));
                }
            });

        [Benchmark]
        public void SumAvxAlignedBufferedSingle() =>
            new AlignedBuffer<float>(fbuffer, a.Length, 32).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void SumAvxAlignedBufferedx1000() =>
            new AlignedBuffer<float>(fbuffer, a.Length, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });

        [Benchmark]
        public void SumAvxAlignedPooledSingle() =>
            new AlignedBuffer<float>(a.Length, 32).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void SumAvxAlignedPooledx1000() =>
            new AlignedBuffer<float>(a.Length, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });

        [Benchmark]
        public void SumAvxAlignedHeapSingle() =>
            new AlignedBuffer<float>(a.Length, 32, false).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void SumAvxAlignedHeapx1000() =>
            new AlignedBuffer<float>(a.Length, 32, false).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });
    }
}
