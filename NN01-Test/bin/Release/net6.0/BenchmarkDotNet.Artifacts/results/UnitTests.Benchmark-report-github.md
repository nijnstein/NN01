``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT  [AttachedDebugger]
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|             Method |         Mean |      Error |     StdDev |
|------------------- |-------------:|-----------:|-----------:|
|             SumAvx |     14.43 ns |   0.315 ns |   0.481 ns |
|            SumFast |     76.68 ns |   1.533 ns |   1.825 ns |
|          SumDotnet |    459.62 ns |   3.549 ns |   3.320 ns |
| SumAvxAlignedx1000 | 15,237.94 ns | 198.302 ns | 185.491 ns |
