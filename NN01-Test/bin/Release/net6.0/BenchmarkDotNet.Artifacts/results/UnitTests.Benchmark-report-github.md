``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT  [AttachedDebugger]
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|                                    Method |         Mean |      Error |     StdDev |
|------------------------------------------ |-------------:|-----------:|-----------:|
| AlignedBuffer_SumAvxAlignedBufferedSingle |     42.50 ns |   0.516 ns |   0.483 ns |
|  AlignedBuffer_SumAvxAlignedBufferedx1000 |     43.87 ns |   0.374 ns |   0.312 ns |
|   AlignedBuffer_SumAvxAlignedPooledSingle |    133.62 ns |   2.707 ns |   3.222 ns |
|    AlignedBuffer_SumAvxAlignedPooledx1000 | 15,199.53 ns | 130.972 ns | 122.512 ns |
|     AlignedBuffer_SumAvxAlignedHeapSingle |    114.64 ns |   1.700 ns |   1.590 ns |
|      AlignedBuffer_SumAvxAlignedHeapx1000 | 14,985.62 ns | 102.463 ns |  90.830 ns |
