using Microsoft.Diagnostics.Tracing.Parsers.AspNet;
using Microsoft.Diagnostics.Tracing.Parsers.ClrPrivate;
using NN01;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Security.Claims;


namespace UnitTests
{
    public class Digit_ocr
    {
        const int ClassCount = 2; 
        int patternSize = 28 * 28;  // 784

        string filename = $"..\\..\\..\\..\\Handwritten Digit Samples\\mnist_nn01.bin"; 


        string[] trainingDataSets = new string[] {
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train0.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train1.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train2.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train3.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train4.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train5.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train6.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train7.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train8.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_train9.jpg",
        };

        string[] testDataSets = new string[] {
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test0.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test1.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test2.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test3.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test4.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test5.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test6.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test7.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test8.jpg",
            "..\\..\\..\\..\\Handwritten Digit Samples\\mnist_test9.jpg",
        };

        public void Run(int steps = 100, bool allowGPU = true)
        {
            Trainer.Settings settings = new Trainer.Settings();
            settings.Population = 12;//64;
            settings.Steps = steps;
            settings.GPU = allowGPU && GPUContext.HaveGPUAcceleration;
            settings.MiniBatchSize = 0;
            settings.LearningRate = 0.01f;
            settings.BatchedLearningRate = 0.03f;
            settings.OnlineTraining = true;
            settings.BatchedStartSteps = 0;
            settings.MutationChance = 0.3f;
            settings.WeightMutationStrength = 0.05f;
            settings.BiasMutationStrength = 0.01f;
            settings.SoftMax = false;
            settings.Random = new CPURandom(RandomDistributionInfo.Uniform(0f, 1f));


            int sampleCountPerClass = 128;
            int testCountPerClass = 64;

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("[L]arge: 4096 / class");
            Console.WriteLine("[M]arge: 1024 / class");
            Console.WriteLine("[S]mall:  512 / class");
            Console.WriteLine("[T]iny:   128 / class= default");
            Console.WriteLine();
            Console.Write("> choose a network size: "); 
            switch(Console.ReadKey().Key)
            {
                case ConsoleKey.L:
                    sampleCountPerClass = 1024 * 4;
                    testCountPerClass = 128 * 16;
                    break;
                case ConsoleKey.M:
                    sampleCountPerClass = 1024;
                    testCountPerClass = 128 * 4;
                    break;
                case ConsoleKey.S:
                    sampleCountPerClass = 512;
                    testCountPerClass = 256;
                    break;
                default: 
                case ConsoleKey.T:
                    sampleCountPerClass = 128;
                    testCountPerClass = 64;
                    break;
            }
            Console.WriteLine();

            Console.WriteLine();
            Console.Write("> number of classes to train:");
            int classCount = int.Parse(Console.ReadLine());
            Console.Write("> number of steps to batch train: ");
            settings.BatchedStartSteps = int.Parse(Console.ReadLine());
            Console.Write("> number of steps to train in total: ");
            settings.Steps = int.Parse(Console.ReadLine());

            settings.ReadyEstimator = (nn) =>
            {
                return (nn.Cost < 0.00001f && nn.Cost != 0);
            };

            //  settings.FitnessEstimator = (network, samples) =>
            //  {
            //       return 1 - Math.Min(0.999999f, network.Cost);
            //   };

            NeuralNetwork nn = new NeuralNetwork(
              new int[]
              {
                              patternSize,
                              28 * 28,
                              28 * 2,
                              classCount,
              },
              new LayerActivationFunction[] {
                                LayerActivationFunction.ReLU,
                                LayerActivationFunction.Tanh,
                               LayerActivationFunction.Sigmoid
              },
              settings.SoftMax,
              settings.Random
            );

            int totalParameterCount = nn.CalculateTotalParameterCount();

            settings.OnStep = (NeuralNetwork network, int step, bool batched, float populationError, float ms, int mutationCount) =>
            {
                if (step == -1)
                {
                    Console.WriteLine($"> Data loaded & prepared in: {ms.ToString("0.000")}ms");
                    Console.WriteLine(">");
                    return;
                }
                if (step == 0)
                {
                    Console.WriteLine($"> Step     Trainingmode       Population Fittness        Lowest Cost      Best Fit         Mutations    %of Population      Steptime     ");
                    Console.WriteLine($"> --------------------------------------------------------------------------------------------------------------------------------------");
                }
                float mp = (100f / totalParameterCount) * ((float)mutationCount / (float)settings.Population);
                Console.WriteLine($"> {(step.ToString().PadRight(8))} {(batched ? "batched      " : "online       ")}      {populationError.ToString("0.00000").PadRight(25)}  {(network.Cost.ToString("0.00000").PadRight(15))}  {(network.Fitness.ToString("0.00000").PadRight(15))}  {mutationCount.ToString().PadRight(12)} {mp.ToString("0.00").PadRight(15)}   {ms.ToString("0.000")}ms");
            };

            Console.WriteLine($" Initializing network");
            Console.WriteLine("");
            Console.WriteLine($">          Structure: {nn.ToString()}");
            Console.WriteLine($">         Input Size: {nn.Input.Size.ToString()}");
            Console.WriteLine($">         Population: {settings.Population}");
            Console.WriteLine($">        Class Count: {nn.Output.Size.ToString()}");
            Console.WriteLine($">    Parameter Count: {totalParameterCount.ToString()}");
            Console.WriteLine($">              Steps: {settings.Steps}");
            Console.WriteLine($">      Training mode: {(settings.OnlineTraining ? "online" : "batched")}");
            Console.WriteLine($">        Batch start: {(settings.BatchedStartSteps > 0 ? $"batching until step {settings.BatchedStartSteps}" : "no")}");
            if (settings.GPU)
            {
                Console.WriteLine($">                GPU: {GPUContext.GetDefaultDeviceDescription()}");
            }
            else
            {
                Console.WriteLine($">                CPU: Multithreaded {Environment.ProcessorCount} cores");
            }
            Console.WriteLine("");

            // train for given patterns 
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.Write("> Loading sample data: ");

            int currentIndex = 0;
            int testIndex = 0;

            SampleSet trainingSet = new SampleSet(28 * 28, sampleCountPerClass * classCount, classCount, settings.Population);
            SampleSet testSet = new SampleSet(28 * 28, testCountPerClass * classCount, classCount, settings.Population);
            for (int i = 0; i < classCount; i++)
            {
                Console.Write($"> {Path.GetFileName(trainingDataSets[i])} ");
                currentIndex = trainingSet.LoadDigitGrid(currentIndex, trainingDataSets[i], i + 1, 28, 28, sampleCountPerClass);
                Console.Write($"> {Path.GetFileName(testDataSets[i])} ");
                testIndex = testSet.LoadDigitGrid(testIndex, testDataSets[i], i + 1, 28, 28, testCountPerClass);
            }
            Console.WriteLine();
            Console.WriteLine();
            sw.Stop();
            trainingSet.Prepare(false);  //  settings.SoftMax);
            testSet.Prepare(false); // settings.SoftMax);

            Console.WriteLine($" loaded {currentIndex} training and {testIndex} test samples in {sw.Elapsed.TotalMilliseconds.ToString("0.000")}ms");
            DisplaySample(trainingSet.probabilityIndex, 28, ConsoleColor.Yellow);


            Console.WriteLine("\n> Starting training.. ");
            sw.Reset();
            sw.Start();

            int stepsTrained = Trainer.Train
                (
                    nn,
                    trainingSet,
                    testSet,
                    settings
                ); 

            Console.WriteLine(); 
            Console.WriteLine($"Training Completed:");
            Console.WriteLine("");
            Console.WriteLine($">          Time: {sw.Elapsed.TotalMilliseconds.ToString("0.000")}ms");
            Console.WriteLine($">      Steptime: {(sw.Elapsed.TotalMilliseconds / (stepsTrained * Trainer.Settings.Default.Population)).ToString("0.000")}ms");
            Console.WriteLine($"> Steps trained: {stepsTrained}");
            Console.WriteLine($">          Cost: {nn.Cost.ToString("0.0000000")}");
            Console.WriteLine($">       Fitness: {nn.Fitness.ToString("0.0000000")}");
            Console.WriteLine("");
            Console.WriteLine("");

            int nOk = 0;
            int nTests = 8; 
            for (int i = 0; i < nTests; i++)
            {
                testIndex = Random.Shared.Next(testSet.SampleCount);
                
                Sample sample = testSet.Samples[testIndex];
                Span<float> outputs = nn.FeedForward(testSet.SampleData(testIndex));
                int index = outputs.ArgMax();

                bool result = index + 1 == sample.Class; 

                Console.WriteLine($"> test {i + 1}: [{testIndex}], class = {sample.Class}, output = {outputs[index]}");
                DisplaySample(testSet.SampleData(testIndex), 28, result ? ConsoleColor.Green : ConsoleColor.Red);

                nOk += result ? 1 : 0; 
            }
            Console.WriteLine($"> {nOk} of {nTests} tests succeeded, {((100f / (float)nTests) * (float)nOk).ToString("0.00")}%");

            try
            {
                using (FileStream fs = File.OpenWrite(filename))
                {
                    nn.WriteTo(fs);
                    Console.WriteLine($"> Network stored at: {filename}");
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString()); 
            }
        }

        void DisplaySample(Span<float> sample, int w, ConsoleColor color)
        {
            //https://en.wikipedia.org/wiki/Box-drawing_character
            var oldColor = Console.ForegroundColor;
            Console.ForegroundColor = color; 

            int c = 0; 
            for (int i = 0; i < sample.Length; i++)
            {
                float f = sample[i];

                c++;
                if (c == w) 
                {
                    c = 0;
                    Console.WriteLine(); 
                }
                if (f < 0.2f)
                {
                    Console.Write(" ");
                }
                else
                if (f < 0.4f)
                {
                    Console.Write(".");
                }
                else
                if (f < 0.6f)
                {
                    Console.Write("_");
                }
                else
                {
                    Console.Write("0");
                }
            }

            Console.ForegroundColor = oldColor; 
        }

    }
}
