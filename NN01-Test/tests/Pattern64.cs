using NN01;
using NSS;
using System.Diagnostics;


namespace UnitTests
{
    public class Pattern64
    {
        int patternSize = 64;

        float[] GetBitPattern(string sample)
        {
            float[] output = new float[patternSize];

            int c = 0;
            for (int i = 0; i < sample.Length && c < patternSize; i++)
            {
                for (int j = 0; j < 6 && c < patternSize; j++, c++)
                {
                    output[c] = (((byte)sample[i] - 32) & (1 << j)) != 0 ? 1f : 0f;
                }
            }
            while (c < patternSize)
            {
                output[c++] = 0f;
            }
            return output;
        }

        public void Run()
        {
            NeuralNetwork nn = new NeuralNetwork(
              new int[]
              {
                  patternSize,
                  48, 24, 1
              },
              new LayerActivationFunction[] {
                    LayerActivationFunction.ReLU,
                    LayerActivationFunction.LeakyReLU,
                    LayerActivationFunction.LeakyReLU,
              }
            );

            Trainer.Settings settings = new Trainer.Settings();
            settings.Population = 1000;
            settings.Steps = 200; 
            settings.ReadyEstimator = (nn) =>
            {
                return nn.Fitness > 0.999f && (nn.Cost < 0.0001f);
            };

            Console.WriteLine($"Training network");
            Console.WriteLine("");
            Console.WriteLine($">          Structure: {nn.ToString()}");
            Console.WriteLine($">         Input Size: {nn.Input.Size.ToString()}");
            Console.WriteLine($">        Class Count: {nn.Output.Size.ToString()}");
            Console.WriteLine($">         Population: {settings.Population}");
            Console.WriteLine($">              Steps: {settings.Steps}");
            Console.WriteLine("");

            // train for given patterns 
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int stepsTrained = Trainer.Train
            (
                    nn,
                    new float[][]
                    {
                        GetBitPattern("JA"),
                        GetBitPattern("YES"),
                        GetBitPattern("JAWEL"),
                        GetBitPattern("OK"),
                        
                        GetBitPattern("TRUE"),
                        GetBitPattern("WAAR"),
                        GetBitPattern("YEAH"),
                        GetBitPattern("1"),
                        
                        GetBitPattern("NEE"),
                        GetBitPattern("NEEN"),
                        GetBitPattern("NO"),
                        GetBitPattern("NOT OK"),
                        
                        GetBitPattern("NOT"),
                        GetBitPattern("FALSE"),
                        GetBitPattern("ONWAAR"),
                        GetBitPattern("NIET"),

                        GetBitPattern("0"),
                   }.ConvertTo2D(),
                    new int[]
                    {
                        1,1,1,1,
                        1,1,1,1, 
                        0,0,0,0,
                        0,0,0,0,
                        0
                    },
                    new float[][]
                    {
                        GetBitPattern("JA"),
                        GetBitPattern("NEE"),
                    }.ConvertTo2D(),
                    new int[]
                    {
                        1, 0
                    },
                    settings
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
            Test(nn, true, "JA", 1);
            Test(nn, true, "OK", 1);
            Test(nn, false, "NON", 0);
            Test(nn, true, "YEAH", 1);
            Test(nn, false, "yes", 1);
        }


        // show results for a single test 
        void Test(NeuralNetwork network, bool inset, string pattern, int classIndex)
        {
            float[] data = GetBitPattern(pattern); 

            ConsoleColor prev = Console.ForegroundColor;
            Span<float> outputs = network.FeedForward(data);
            float output = outputs[0];

            Console.Write($"Test    {(inset ? "[in training set]" : "[in test set]").PadRight(24)} {pattern.PadRight(32)} == {classIndex} => ");

            if (classIndex == 1 && output > 0.5f)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"OK {Math.Min(100f, Math.Abs(output) * 100f).ToString("0")}%");
            }
            else
            if (classIndex == 0 && output < 0.5f)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"OK {Math.Min(100f, Math.Abs(1f - output) * 100f).ToString("0")}%");
            }
            else
            {
                if (classIndex == 0) output = 1f - output; 
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Test    {(inset ? "[in training set]": "[in test set]").PadRight(24)} {pattern.PadRight(32)} == {classIndex} => FAIL {(output * 100f).ToString("0")}%");
            }

            
            Console.ForegroundColor = prev;
        }
    }
}
