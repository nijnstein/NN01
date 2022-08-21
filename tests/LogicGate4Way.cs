using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01.Tests
{
    public class LogicGate4Way
    {
        const int patternCount = 16;
        const int testPatternCount = 6;
        const int classCount = 1;   // actually its 2, there is always class 0 -> unclassified 

        float[][] trainingPatterns = new float[patternCount][]
        {
            new float[] { 0, 0, 0, 0 },
            new float[] { 0, 0, 0, 1 },
            new float[] { 0, 0, 1, 0 },
            new float[] { 0, 0, 1, 1 },
            new float[] { 0, 1, 0, 0 },
            new float[] { 0, 1, 0, 1 },
            new float[] { 0, 1, 1, 0 },
            new float[] { 0, 1, 1, 1 },
            new float[] { 1, 0, 0, 0 },
            new float[] { 1, 0, 0, 1 },
            new float[] { 1, 0, 1, 0 },
            new float[] { 1, 0, 1, 1 },
            new float[] { 1, 1, 0, 0 },
            new float[] { 1, 1, 0, 1 },
            new float[] { 1, 1, 1, 0 },
            new float[] { 1, 1, 1, 1 },
        };

        int[] trainingClasses = new int[patternCount]
        {
            0, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 0
        };

        float[][] testPatterns = new float[testPatternCount][]
        {
            new float[] { -0.1f, 0, 0, 0.2f },
            new float[] { 1.2f, 1, 0, 1 },
            new float[] { 1, 1, 1.13f, 0 },
            new float[] { 1, 1.1f, 1, 0.8f },
            new float[] { 1, 0.1f, -0.3f, 1.01f },
            new float[] { 0.9f, 1.2f, 1.04f, 0.8f },
        };

        int[] testClasses = new int[testPatternCount]
        {
            0, 1, 1, 0, 1, 0
        };

        public void Run()
        {
            NeuralNetwork nn = new NeuralNetwork(
                new int[] { 4, 16, 8, classCount },
                new LayerActivationFunction[] {
                    LayerActivationFunction.ReLU,
                    LayerActivationFunction.LeakyReLU,// Swish,
                    LayerActivationFunction.LeakyReLU,
                }
            );

            Console.WriteLine($"Training network for 4-gate something");
            Console.WriteLine("");
            Console.WriteLine($">          Structure: {nn.ToString()}");
            Console.WriteLine($">         Input Size: {nn.Input.Size.ToString()}");
            Console.WriteLine($">        Class Count: {nn.Output.Size.ToString()}");
            Console.WriteLine($">         Population: {Trainer.Settings.Default.Population}");
            Console.WriteLine($">              Steps: {Trainer.Settings.Default.Steps}");
            Console.WriteLine($">  Training patterns: {trainingPatterns.Length}");
            Console.WriteLine($">      Test patterns: {testPatterns.Length}");
            Console.WriteLine("");
            Console.WriteLine("");

            // train for given patterns 
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int stepsTrained = Trainer.Train
            (
                    nn,
                    trainingPatterns,
                    trainingClasses,
                    testPatterns,
                    testClasses
            );

            sw.Stop();

            Console.WriteLine($"Training Completed:");
            Console.WriteLine("");
            Console.WriteLine($">          Time: {sw.Elapsed.TotalMilliseconds.ToString("0.000")}ms");
            Console.WriteLine($">      Steptime: {(sw.Elapsed.TotalMilliseconds / (stepsTrained * Trainer.Settings.Default.Population)).ToString("0.000")}ms");
            Console.WriteLine($"> Steps trained: {stepsTrained}");
            Console.WriteLine($">          Cost: {nn.Cost.ToString("0.0000000")}");
            Console.WriteLine($">       Fitness: {nn.Fitness.ToString("0.0000000")}");
            Console.WriteLine("");
            Console.WriteLine("");

            // test patterns 
            for (int i = 0; i < testPatternCount; i++)
            {
                Test(nn, string.Join(' ', testPatterns[i]).PadRight(32), testPatterns[i], testClasses[i]);
            }
        }


        // show results for a single test 
        void Test(NeuralNetwork network, string msg, float[] data, float label)
        {
            bool IsSet(float value) => value > 0.6f;

            ConsoleColor prev = Console.ForegroundColor;
            float output = network.FeedForward(data)[0];
            if (IsSet(label) == IsSet(output))
            {
                Console.Write($"Test {msg} == {label} => ");
                Console.ForegroundColor = ConsoleColor.Green;

                if (!IsSet(label)) output = 1f - Math.Abs(output);

                Console.WriteLine($"OK {Math.Min(100f, Math.Abs(output) * 100f).ToString("0")}%");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Test    {msg} == {label} => FAIL {(output * 100f).ToString("0")}%");
            }
            Console.ForegroundColor = prev;
        }

        void Reflect(NeuralNetwork network, float[] data)
        {
            float[] output = network.FeedBackward(network.FeedForward(data));

            Console.WriteLine($"Reflect {string.Join(' ', data).PadRight(32)} == {string.Join(' ', output).PadLeft(32)}");

        }
    }
}
