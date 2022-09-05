``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT DEBUG
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|                       Method |           Mean |         Error |        StdDev |
|----------------------------- |---------------:|--------------:|--------------:|
| GPURandom1000xSingleGaussian |       1.717 μs |     0.0222 μs |     0.0197 μs |
|  GPURandom1000xBlockGaussian |   8,856.237 μs |   173.7065 μs |   231.8932 μs |
|       CPURandom1000xGaussian |       8.042 μs |     0.0374 μs |     0.0332 μs |
|  CPURandom1000xBlockGaussian | 228,096.721 μs | 1,956.6179 μs | 1,734.4908 μs |
