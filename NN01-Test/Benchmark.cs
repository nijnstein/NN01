using BenchmarkDotNet.Attributes;
using NN01;
using NUnit.Framework;
using System.Buffers;

namespace UnitTests
{

    public class Benchmark
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

                for(int i = 0; i < s.Length; i++)
                {
                    s[i] = 2f; 
                }
            });
        }


// |    Method |      Mean |    Error |   StdDev |
// |---------- |----------:|---------:|---------:|
// |    MaxAVX |  12.95 ns | 0.116 ns | 0.103 ns |
// |   MaxFast |  84.46 ns | 0.313 ns | 0.277 ns |
// | MaxDotnet | 483.69 ns | 2.373 ns | 2.220 ns |
            
//        [Benchmark] public float MaxAVX() => Intrinsics.Max(a);
//        [Benchmark] public float MaxFast() => Intrinsics.MaxUnaligned(a);
//        [Benchmark] public float MaxDotnet() => a.Max();


//|             Method |         Mean |      Error |     StdDev |
//|------------------- |-------------:|-----------:|-----------:|
//|             SumAvx |     14.43 ns |   0.315 ns |   0.481 ns |
//|            SumFast |     76.68 ns |   1.533 ns |   1.825 ns |
//|          SumDotnet |    459.62 ns |   3.549 ns |   3.320 ns |
//| SumAvxAlignedx1000 | 15,237.94 ns | 198.302 ns | 185.491 ns |         looks like there is no penalty for misalignment on this machine..
        //[Benchmark] public float SumAvx() => Intrinsics.Sum(a);
        //[Benchmark] public float SumFast() => Intrinsics.SumUnaligned(a);
        //[Benchmark] public float SumDotnet() => a.Sum();
        //[Benchmark] public void SumAvxAlignedx1000() => 
        //    buffer.With((lease) => 
        //    {
        //        for (int i = 0; i < 1000; i++)
        //        {
        //            Intrinsics.Sum(lease.GetSpan(0, a.Length));
        //        }
        //    });

        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedBufferedSingle() =>
            new AlignedBuffer<float>(fbuffer, a.Length, 32).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedBufferedx1000() =>
            new AlignedBuffer<float>(fbuffer, a.Length, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });

//|   AlignedBuffer_SumAvxAlignedPooledSingle |    133.62 ns |   2.707 ns |   3.222 ns |
//|    AlignedBuffer_SumAvxAlignedPooledx1000 | 15,199.53 ns | 130.972 ns | 122.512 ns |
//|     AlignedBuffer_SumAvxAlignedHeapSingle |    114.64 ns |   1.700 ns |   1.590 ns |
//|      AlignedBuffer_SumAvxAlignedHeapx1000 | 14,985.62 ns | 102.463 ns |  90.830 ns |
/*
        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedPooledSingle() =>
            new AlignedBuffer<float>(a.Length, 32).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedPooledx1000() =>
            new AlignedBuffer<float>(a.Length, 32).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });

        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedHeapSingle() =>
            new AlignedBuffer<float>(a.Length, 32, false).With((lease) =>
            {
                Intrinsics.Sum(lease.GetSpan(0, a.Length));
            });

        [Benchmark]
        public void AlignedBuffer_SumAvxAlignedHeapx1000() =>
            new AlignedBuffer<float>(a.Length, 32, false).With((lease) =>
            {
                Span<float> s = lease.GetSpan(0, a.Length);
                for (int i = 0; i < 1000; i++) Intrinsics.Sum(s);
            });
*/

        // | ExpAvxFast |  41.53 ns | 0.289 ns | 0.270 ns |
        // | Exp        | 453.79 ns | 3.111 ns | 2.758 ns |
        //       [Benchmark] public Span<float> ExpAvxFast() => MathEx.ExpFast(a);
        //       [Benchmark] public Span<float> Exp() => MathEx.Exp(a);




    }
}
