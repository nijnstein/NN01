using BenchmarkDotNet.Attributes;
using NN01;

namespace UnitTests
{
    
    public class Benchmark
    {
        public float[] a = new float[]
        {
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001,
            12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 12.5f, 12.1f, 1f, 0f, 1, 3, 4, 5, 6, 7, 87, 4, 5, 1001
        };

        // |    Method |       Mean |     Error |    StdDev |
        // |---------- |-----------:|----------:|----------:|
        // |    MaxAVX |   7.401 ns | 0.0676 ns | 0.0632 ns |
        // | MaxDotnet | 178.679 ns | 0.7756 ns | 0.7255 ns |

        [Benchmark] public float MaxAVX() => a.AsSpan().Max();
        [Benchmark] public float MaxDotnet() => a.Max();


    }
}
