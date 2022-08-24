﻿using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Validators;

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
            Console.WriteLine("Press [B] to benchmark, any other for test run.");
            if (Console.ReadKey().Key == ConsoleKey.B)
            {
                BenchmarkRunner.Run<Benchmark>(new AllowNonOptimized());
            }
            else
            {
                do
                {
                    new LogicGate4Way().Run();
                    Console.WriteLine("");
                    Console.WriteLine("");

                    new Pattern64().Run();

                    Console.WriteLine("");
                    Console.WriteLine("");

                    Console.WriteLine("[SPACEBAR] to run tests again, any other to exit");
                    Console.WriteLine("");
                    Console.WriteLine("");
                }
                while (Console.ReadKey().Key == ConsoleKey.Spacebar);
            }
        }
    }
}