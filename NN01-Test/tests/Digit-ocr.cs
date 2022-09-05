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
        string filename = "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_nn01.bin"; 


        string[] trainingDataSets = new string[] {
            "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train0.jpg",
            "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train1.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train2.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train3.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train4.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train5.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train6.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train7.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train8.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_train9.jpg",
        };

        string[] testDataSets = new string[] {
            "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test0.jpg",
            "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test1.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test2.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test3.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test4.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test5.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test6.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test7.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test8.jpg",
 //           "C:\\Repo\\NN01\\NN01\\Handwritten Digit Samples\\mnist_test9.jpg",
        }; 

        public void Run(int steps = 100, bool allowGPU = true)
        {
            NeuralNetwork nn = new NeuralNetwork(
              new int[]
              {
                  patternSize,
                  28 * 14,
                  32,
                  ClassCount
              },
              new LayerActivationFunction[] {
                    LayerActivationFunction.ReLU,
                    LayerActivationFunction.Tanh,//Swish, // .Tanh,
                    LayerActivationFunction.LeakyReLU,
              }
            );

            int sampleCountPerClass = 256;
            int testCountPerClass = 16;

            Trainer.Settings settings = new Trainer.Settings();
            settings.Population = 48;
            settings.Steps = steps;
            settings.GPU = allowGPU && GPUContext.HaveGPUAcceleration;
            settings.MiniBatchSize = 0;
            settings.LearningRate = 0.05f;
            settings.OneByOne = true;
            settings.ReadyEstimator = (nn) =>
            {
                return  (nn.Cost < 0.001f && nn.Cost != 0);
            };
            settings.FitnessEstimator = (network, samples) =>
            {
                return 1 - Math.Min(0.999999f, network.Cost);
            };
            settings.OnStep = (NeuralNetwork network, int step) => Console.WriteLine($"> Step: {(step.ToString().PadRight(10))} Cost: {(network.Cost.ToString("0.00000").PadRight(10))} Fitness:  {(network.Fitness.ToString("0.00000").PadRight(10))}");


            Console.WriteLine($" Initializing network");
            Console.WriteLine("");
            Console.WriteLine($">          Structure: {nn.ToString()}");
            Console.WriteLine($">         Input Size: {nn.Input.Size.ToString()}");
            Console.WriteLine($">        Class Count: {nn.Output.Size.ToString()}");
            Console.WriteLine($">         Population: {settings.Population}");
            Console.WriteLine($">              Steps: {settings.Steps}");
            if(settings.GPU)
            {
                Console.WriteLine($">                GPU: {"enabled"}");
            }
            Console.WriteLine("");

            // train for given patterns 
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.WriteLine("> Loading samples: ");

            int currentIndex = 0;
            int testIndex = 0;

            SampleSet trainingSet = new SampleSet(28 * 28, sampleCountPerClass * ClassCount, ClassCount);
            for (int i = 0; i < ClassCount; i++)
            {
                Console.Write($"> {Path.GetFileName(trainingDataSets[i])} ");
          
                currentIndex = trainingSet.LoadDigitGrid(currentIndex, trainingDataSets[i], i + 1, 28, 28, sampleCountPerClass);
            }
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($" loaded {currentIndex} training samples");
            Console.WriteLine();

            SampleSet testSet = new SampleSet(28 * 28, testCountPerClass * ClassCount, ClassCount);
            for (int i = 0; i < ClassCount; i++)
            {
                Console.Write($"> {Path.GetFileName(testDataSets[i])} ");
                testIndex = testSet.LoadDigitGrid(testIndex, testDataSets[i], i + 1, 28, 28, testCountPerClass); 
            }
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($" loaded {testIndex} test samples");
            Console.WriteLine();

            trainingSet.Prepare(); 
            testSet.Prepare(); 

            sw.Stop();
            Console.WriteLine($"> Data ready in: {sw.Elapsed.TotalMilliseconds.ToString("0.000")}ms");
            Console.WriteLine();
            DisplaySample(trainingSet.probabilityIndex, 28, ConsoleColor.Yellow);
            Console.WriteLine();


            Console.WriteLine("\n> Starting training..");
            sw.Reset();
            sw.Start(); 

            int stepsTrained = Trainer.Train
            (
                nn,
                trainingSet,
                testSet,
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
