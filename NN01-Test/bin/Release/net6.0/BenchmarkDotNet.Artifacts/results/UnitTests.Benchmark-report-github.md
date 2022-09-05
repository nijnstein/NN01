``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT  [AttachedDebugger]
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|                          Method |          Mean |         Error |         StdDev |        Median |
|-------------------------------- |--------------:|--------------:|---------------:|--------------:|
|              GPURandom1MxSingle |      1.982 μs |     0.0289 μs |      0.0256 μs |      1.977 μs |
|      GPURandom1000xSingleNormal |      2.976 μs |     0.0557 μs |      0.0521 μs |      2.961 μs |
|    GPURandom1000xSingleGaussian |      4.823 μs |     0.3989 μs |      1.1761 μs |      5.389 μs |
|           GPURandom1MxBlockSpan | 19,411.628 μs |   366.3882 μs |    342.7198 μs | 19,418.645 μs |
|   GPURandom1000xBlockSpanNormal | 85,024.736 μs | 1,706.8877 μs |  4,758.1207 μs | 85,107.850 μs |
| GPURandom1000xBlockSpanGaussian | 87,378.185 μs | 4,875.1305 μs | 14,374.4323 μs | 82,886.991 μs |
