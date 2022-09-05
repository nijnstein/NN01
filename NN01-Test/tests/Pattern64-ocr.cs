using Microsoft.Diagnostics.Tracing.Parsers.ClrPrivate;
using NN01;
using NSS;
using System.Diagnostics;
using System.Security.Claims;


namespace UnitTests
{
    public class Pattern64_ocr
    {
        int patternSize = 64;

        string[] patternStrings = new string[]{
// 0
"11111111" +
"10000001" +
"10000001" +
"10000001" +
"10000001" +
"10000001" +
"10000001" +
"11111111",
// 1
"00011000" +
"00011000" +
"00011000" +
"00011000" +
"00011000" +
"00011000" +
"00011000" +
"00011000",
// 2
"11111111" +
"00000001" +
"00000001" +
"11111111" +
"10000000" +
"10000000" +
"10000000" +
"01111110",
// 3
"11111111" +
"00000001" +
"00000001" +
"00000001" +
"11111111" +
"00000001" +
"00000001" +
"11111111",
// 4
"01000010" +
"01000010" +
"11000010" +
"10000010" +
"11111111" +
"00000010" +
"00000010" +
"00000010",
// 5            
"01111111" +
"10000000" +
"10000000" +
"10000000" +
"01111110" +
"00000001" +
"10000001" +
"01111110",
// 6            
"00100000" +
"01000000" +
"01000000" +
"10000000" +
"11111110" +
"10000001" +
"10000001" +
"01111110",
// 7            
"11111111" +
"00000010" +
"00000100" +
"00001000" +
"00010000" +
"00100000" +
"01000000" +
"10000000",
// 8
"01111110" +
"10000001" +
"10000001" +
"01111110" +
"10000001" +
"10000001" +
"10000001" +
"01111110",
// 9
"01111110" +
"10000001" +
"10000001" +
"01111110" +
"00000010" +
"00000001" +
"00000001" +
"01111110"
            };



        string[] testStrings = new string[]{
// 0
"11111111" +
"11000001" +
"10000001" +
"10000001" +
"10000001" +
"11000001" +
"11000001" +
"11111110",
// 1
"00010000" +
"00010000" +
"00010000" +
"00010000" +
"00010000" +
"00010000" +
"00010000" +
"00010000",
// 2
"11111111" +
"00000011" +
"00000011" +
"00000001" +
"11111111" +
"11000000" +
"11000000" +
"01111110",
// 3
"11111111" +
"00000001" +
"00000001" +
"11111111" +
"00000000" +
"00000001" +
"00000001" +
"11111111",
// 4
"10000010" +
"10000010" +
"10000010" +
"11111110" +
"00000010" +
"00000010" +
"00000010" +
"00000010",
// 5            
"01111111" +
"10000000" +
"10000000" +
"11111100" +
"00000010" +
"00000001" +
"10000001" +
"01111110",
// 6            
"00100000" +
"01000000" +
"01000000" +
"11111110" +
"10000010" +
"10000001" +
"10000001" +
"01111110",
// 7            
"00111111" +
"00000010" +
"00000100" +
"00001000" +
"00010000" +
"00100000" +
"00100000" +
"00100000",
// 8
"01111110" +
"10000001" +
"10000001" +
"10000011" +
"01111110" +
"10000001" +
"10000001" +
"01111110",
// 9
"01111110" +
"10000001" +
"10000001" +
"10000001" +
"01111110" +
"00000001" +
"00000001" +
"01111110"
            };


        float[] GetPattern(string sample)
        {
            float[] output = new float[patternSize];

            for (int i = 0; i < patternSize; i++)
            {
                output[i] = sample[i] == '0' ? 0f : 1f;
            }
            return output;
        }

        float[] Distort(float[] sample, float chance = 1 / 16f)
        {
            for (int i = 0; i < sample.Length; i++)
            {
                if(Random.Shared.NextSingle() < chance)
                {
                    sample[i] = sample[i] > 0.5 ? 0f : 1f;
                }
            }
            return sample;
        }

        public void Run(int steps = 20)
        {
            NeuralNetwork nn = new NeuralNetwork(
              new int[]
              {
                  patternSize,
                  64,
                  32,
                  10
              },
              new LayerActivationFunction[] {
                    LayerActivationFunction.ReLU,
                    LayerActivationFunction.Tanh,
                    LayerActivationFunction.LeakyReLU,
              }
            );

            Trainer.Settings settings = new Trainer.Settings();
            settings.Population = 100;
            settings.Steps = steps;
            settings.GPU = false; 
            settings.ReadyEstimator = (nn) =>
            {
                return /* (nn.Cost > 0 && nn.CostDelta < 0.000001) || */ (nn.Fitness > 0.99f && nn.Cost < 0.005f);
            };
         //   settings.OnStep = (NeuralNetwork network, int step) => Console.WriteLine($"> Step: {(step.ToString().PadRight(10))} Cost: {(network.Cost.ToString("0.0000").PadRight(10))} Fitness:  {(network.Fitness.ToString("0.0000").PadRight(10))}");


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

            List<float[]> patterns = new List<float[]>();
            patterns.AddRange(patternStrings.Select(x => GetPattern(x).ToArray()));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 32f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 32f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 16f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 8f)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 0)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 32f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 16f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 8f)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 16f)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 32f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 16f)));
            patterns.AddRange(patternStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 8f)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 16f)));
            patterns.AddRange(testStrings.Select(x => Distort(GetPattern(x).ToArray(), 1 / 32f)));

            List<int> classes = new List<int>();
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            classes.AddRange(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

            int stepsTrained = Trainer.Train
            (
                    nn,
                    patterns.ToArray().ConvertTo2D(),
                    classes.ToArray(),
                    testStrings.Select(x => GetPattern(x).ToArray()).ToArray().ConvertTo2D(),
                    new int[]
                    {
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
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
            Test(nn, true, GetPattern(patternStrings[0]).ToArray(), 1);
            Test(nn, false, Distort(GetPattern(patternStrings[0]).ToArray(), 1 / 16f), 1);

            Test(nn, false, GetPattern(testStrings[0]).ToArray(), 1);
            Test(nn, false, Distort(GetPattern(testStrings[0]).ToArray(), 1 / 32f), 1);
        }
        // Training network
        // 
        // >          Structure: [64-ReLU-256-Tanh-128-LeakyReLU-10]
        // >         Input Size: 64
        // >        Class Count: 10
        // >         Population: 1000
        // >              Steps: 500
        // 
        // Training Completed:
        // 
        // >          Time: 237112,187ms
        // >      Steptime: 4,742ms         with AVX -> 2.2 ms :)
        // > Steps trained: 500
        // >          Cost: 0,0042749
        // >       Fitness: 0,9672515
        // 
        // Test[in training set]        1 => OK 89%

        // show results for a single test 
        void Test(NeuralNetwork network, bool inset, float[] pattern, int classIndex)
        {
            ConsoleColor prev = Console.ForegroundColor;
            Span<float> outputs = network.FeedForward(pattern);


            int iclass = outputs.ArgMax();
            float output = iclass >= 0 ? outputs[classIndex - 1] : float.NaN;

            Console.Write($"Test    {(inset ? "[in training set]" : "[in test set]").PadRight(24)} {classIndex} => ");

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

            int c = 0; 
            foreach(float f in pattern)
            {
                c++;
                Console.Write(f > 0 ? '1' : '0');
                if (c == 8)
                {
                    Console.WriteLine();
                    c = 0;
                }
            }

            Console.ForegroundColor = prev;
        }
    }
}
