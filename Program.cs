﻿using NN01;
using System.Diagnostics;
using System.Linq.Expressions;


void Test(NeuralNetwork network, string msg, float[] data, float label)
{
    bool IsSet(float value) => value > 0.6f;

    ConsoleColor prev = Console.ForegroundColor;
    float output = network.FeedForward(data)[0];
    if (IsSet(label) == IsSet(output))
    {
        Console.Write($"Test {msg} == {label} => ");
        Console.ForegroundColor = ConsoleColor.Green;

        if (!IsSet(label)) output = output + 1f; 

        Console.WriteLine($"OK {(output * 100f).ToString("0")}%"); 
    }
    else
    {
        Console.ForegroundColor = ConsoleColor.Red; 
        Console.WriteLine($"Test {msg} == {label} => FAIL {(output * 100f).ToString("0")}%");
    }
    Console.ForegroundColor = prev;
}


const int populationCount = 100;

const int patternCount = 16;
const int testPatternCount = 6;

const int classCount = 1;   // actually its 2, there is always class 0 -> unclassified 
const int steps = 10000;


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


do
{
    NeuralNetwork nn = new NeuralNetwork(
        new int[] { 4, 16, classCount },
        new LayerActivationFunction[] {
            LayerActivationFunction.ReLU,
            LayerActivationFunction.LeakyReLU,
        }
    );

    Console.WriteLine($"Training network for 4-gate XOR");
    Console.WriteLine("");
    Console.WriteLine($">          Structure: {nn.ToString()}");
    Console.WriteLine($">         Input Size: {nn.Input.Size.ToString()}");
    Console.WriteLine($">        Class Count: {nn.Output.Size.ToString()}");
    Console.WriteLine($">         Population: {populationCount}");
    Console.WriteLine($">              Steps: {steps}");
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
            testClasses,
            (cost, fitness) =>
            {
                return fitness > 0.999f && (cost < 0.01f);
            },
            steps, 
            populationCount
    ); 

    sw.Stop();

    Console.WriteLine($"Training Completed:");
    Console.WriteLine("");
    Console.WriteLine($">          Time: {sw.Elapsed.TotalMilliseconds.ToString("0.000")}ms");
    Console.WriteLine($">      Steptime: {(sw.Elapsed.TotalMilliseconds / (stepsTrained * populationCount)).ToString("0.000")}ms");
    Console.WriteLine($"> Steps trained: {stepsTrained}");
    Console.WriteLine($">          Cost: {nn.cost.ToString("0.0000000")}");
    Console.WriteLine($">       Fitness: {nn.fitness.ToString("0.0000000")}");
    Console.WriteLine("");
    Console.WriteLine("");

    // test patterns 
    for (int i = 0; i < testPatternCount; i++)
    {
        Test(nn, string.Join(' ', testPatterns[i]).PadRight(32), testPatterns[i], testClasses[i]); 
    }

    Console.WriteLine("");
    Console.WriteLine("[SPACEBAR] to run tests again, any other to exit");
}
while (Console.ReadKey().Key == ConsoleKey.Spacebar);