using NN01;
using NSS;
using System.Diagnostics;


namespace UnitTests
{
    public class Pattern64_multiclass
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
                  48,
                  8,
                  2
              },
              new LayerType[] {
                    LayerType.ReLU,
                    LayerType.Swish,
                    LayerType.LeakyReLU,
              }
            );

            Trainer.Settings settings = new Trainer.Settings();
            settings.Population = 100;
            settings.Steps = 1000; 
            settings.ReadyEstimator = (nn) =>
            {
                return nn.Fitness > 0.99f && (nn.Cost < 0.01f);
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

            int stepsTrained =  Trainer.Train
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
                        2,2,2,2,
                        2,2,2,2,
                        2
                    },
                    (new float[][]
                    {
                        GetBitPattern("ja"),
                        GetBitPattern("YES"),
                        GetBitPattern("JaWel"),
                        GetBitPattern("ok"),

                        GetBitPattern("true"),
                        GetBitPattern("waaR"),
                        GetBitPattern("YEAH"),
                        GetBitPattern("1"),

                        GetBitPattern("NEE"),
                        GetBitPattern("NEEN"),
                        GetBitPattern("no"),
                        GetBitPattern("NOT OK"),

                        GetBitPattern("not"),
                        GetBitPattern("false"),
                        GetBitPattern("ONWAAR"),
                        GetBitPattern("NIET"),

                        GetBitPattern("0"),
                        GetBitPattern("NON")
                    }).ConvertTo2D(),
                    new int[]
                    {
                        1,1,1,1,
                        1,1,1,1,
                        2,2,2,2,
                        2,2,2,2,
                        2,2
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
            Test(nn, false, "NON", 2);
            Test(nn, true, "YEAH", 1);
            Test(nn, false, "yes", 1);
            Test(nn, false, "nee", 2);
            Test(nn, false, "NEE", 2);
        }


        // show results for a single test 
        void Test(NeuralNetwork network, bool inset, string pattern, int classIndex)
        {
            float[] data = GetBitPattern(pattern); 

            ConsoleColor prev = Console.ForegroundColor;
            Span<float> outputs = network.FeedForward(data);


            int iclass = outputs.ArgMax();
            float output = iclass >= 0 ? outputs[iclass] : float.NaN;

            Console.Write($"Test    {(inset ? "[in training set]" : "[in test set]").PadRight(24)} {pattern.PadRight(32)} == {classIndex} => ");

            if (classIndex - 1 == iclass)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"OK {Math.Min(100f, Math.Abs(output) * 100f).ToString("0")}%");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"FAIL {Math.Min(100f, Math.Abs(output) * 100f).ToString("0")}%");
            }


            Console.ForegroundColor = prev;
        }
    }
}
