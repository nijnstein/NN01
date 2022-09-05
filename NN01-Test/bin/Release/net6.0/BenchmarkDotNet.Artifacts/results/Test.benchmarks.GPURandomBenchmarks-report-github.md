``` ini

BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19044.1889 (21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=7.0.100-preview.5.22307.18
  [Host]     : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT
  DefaultJob : .NET 6.0.8 (6.0.822.36306), X64 RyuJIT


```
|                          Method |           Mean |         Error |         StdDev |         Median |
|-------------------------------- |---------------:|--------------:|---------------:|---------------:|
|              GPURandom1MxSingle |       1.648 μs |     0.0182 μs |      0.0152 μs |       1.643 μs |
|      GPURandom1000xSingleNormal |       2.619 μs |     0.0523 μs |      0.0537 μs |       2.619 μs |
|    GPURandom1000xSingleGaussian |       4.524 μs |     0.2269 μs |      0.6174 μs |       4.681 μs |
|               GPURandom1MxBlock |  30,711.682 μs | 1,609.8979 μs |  4,721.5520 μs |  31,859.894 μs |
|       GPURandom1000xBlockNormal |  46,426.224 μs |   923.9546 μs |  1,568.9450 μs |  45,834.400 μs |
|     GPURandom1000xBlockGaussian |  77,321.317 μs | 1,828.9576 μs |  5,247.6223 μs |  77,627.518 μs |
|           GPURandom1MxBlockSpan |  29,669.698 μs | 1,001.2490 μs |  2,807.6096 μs |  30,434.791 μs |
|   GPURandom1000xBlockSpanNormal |  44,290.046 μs |   728.0281 μs |    680.9979 μs |  44,276.175 μs |
| GPURandom1000xBlockSpanGaussian |  82,843.162 μs | 5,152.4410 μs | 15,192.0886 μs |  77,459.675 μs |
|              CPURandom1MxSingle |       4.199 μs |     0.0377 μs |      0.0352 μs |       4.190 μs |
|            CPURandom1000xNormal |       4.189 μs |     0.0380 μs |      0.0337 μs |       4.184 μs |
|          CPURandom1000xGaussian |       8.013 μs |     0.0755 μs |      0.0669 μs |       8.027 μs |
|               CPURandom1MxBlock |  75,744.265 μs |   791.2895 μs |    740.1727 μs |  75,739.243 μs |
|       CPURandom1000xBlockNormal | 230,497.260 μs | 2,167.4831 μs |  1,921.4173 μs | 230,264.267 μs |
|     CPURandom1000xBlockGaussian | 463,071.553 μs | 3,019.6801 μs |  2,824.6107 μs | 462,248.800 μs |
