``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT  [AttachedDebugger]
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|    Method |      Mean |    Error |   StdDev |
|---------- |----------:|---------:|---------:|
|    MaxAVX |  12.42 ns | 0.184 ns | 0.172 ns |
| MaxDotnet | 514.38 ns | 4.360 ns | 3.865 ns |
