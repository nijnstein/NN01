using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Validators;
using NN01;
using System.Runtime.CompilerServices;
using Test;

namespace UnitTests
{
    internal class Program
    {
        public class AllowNonOptimized : ManualConfig
        {
            public AllowNonOptimized()
            {
                Add(JitOptimizationsValidator.DontFailOnError); // ALLOW NON-OPTIMIZED DLLS
                Add(DefaultConfig.Instance.GetLoggers().ToArray()); // manual config has no loggers by default
                Add(DefaultConfig.Instance.GetExporters().ToArray()); // manual config has no exporters by default
                Add(DefaultConfig.Instance.GetColumnProviders().ToArray()); // manual config has no columns by default
            }
        }

        static void Main(string[] args)
        {
            do
            {
                Console.WriteLine("Press [B] to benchmark, any other for test run.");
                if (Console.ReadKey().Key == ConsoleKey.B)
                {
                    //BenchmarkRunner.Run<Benchmark>(new AllowNonOptimized());
                    //BenchmarkRunner.Run<AlignedBufferBenchmark>(new AllowNonOptimized());
                    //BenchmarkRunner.Run<IntrinsicsBenchmark>(new AllowNonOptimized());
                    BenchmarkRunner.Run<GPURandomBenchmarks>(new AllowNonOptimized());
                }
                else
                {
                   //   new LogicGate4Way().Run();
                    Console.WriteLine("");
                    Console.WriteLine("");

                   //   new Pattern64().Run();

                    Console.WriteLine("");
                    Console.Write("Use GPU? [Y/N]");

                    bool gpu = Console.ReadKey().Key == ConsoleKey.Y;

                    new Digit_ocr().Run(25, gpu);










                    // new Pattern64_cpu().Run();

                    // Console.WriteLine("");
                    //Console.WriteLine("");

                    // new Pattern64_multiclass().Run();

                    //  Console.WriteLine("");
                    //   Console.WriteLine("");


                    // new Pattern64_ocr().Run(1000);

                    //   Console.WriteLine("");
                    //   Console.WriteLine("");
                    //
                    // try
                    //// {
                    ////     new Pattern64_ocr_gpu().Run(2, true);
                    //// }
                    //// catch (Exception ex)
                    //// {
                    ////     Console.WriteLine(ex.ToString());
                    //// }
                    ////
                    //// Console.WriteLine("");
                    ////
                    //// Console.WriteLine("");
                    //// try
                    //// {
                    ////     new Pattern64_ocr_gpu().Run(2, false);
                    //// }
                    //// catch (Exception ex)
                    //// {
                    //     Console.WriteLine(ex.ToString());
                    // }

                    Console.WriteLine("");
                    Console.WriteLine("");
                }
            }
            while (Console.ReadKey().Key == ConsoleKey.Spacebar);
        }
    }
}