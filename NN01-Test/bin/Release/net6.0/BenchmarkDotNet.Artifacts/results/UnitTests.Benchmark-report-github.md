``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT  [AttachedDebugger]
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|    Method |      Mean |    Error |   StdDev |
|---------- |----------:|---------:|---------:|
|    SumAvx |  14.43 ns | 0.179 ns | 0.167 ns |
|   SumFast |  76.11 ns | 1.510 ns | 1.551 ns |
| SumDotnet | 458.25 ns | 5.504 ns | 5.149 ns |
