# NN01
My first neural since computers became fast :) 

# Architecture
NN01 uses a classic backpropagating neural network with multiple hidden layers

# Usage
NN01 can be used to classify 1D sequences:

```csharp
      
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
        
       // create a network
       NeuralNetwork nn = new NeuralNetwork(
            new int[] 
            { 
                4,  // 4 inputs
                16, // 16 + 8 hidden
                8, 
                1   // 1 output
            },
            new LayerActivationFunction[] 
            {
                LayerActivationFunction.ReLU,
                LayerActivationFunction.LeakyReLU, // Swish,
                LayerActivationFunction.LeakyReLU,
            }
       );
            
       // train it
       Trainer.Train
       (
            nn,
            trainingPatterns,
            trainingClasses,
            testPatterns,
            testClasses
       );

       // test if class 1 
       bool isClass1 = network.FeedForward(testPattern[0])[0] > 0.5f; 

```

# Fitness evaluation can be modified to suit environment
```csharp
       NeuralNetwork nn = new NeuralNetwork(
           new int[]
           { 
               patternSize,
               32, 16, 1
           },
           new LayerActivationFunction[] {
               LayerActivationFunction.ReLU,
               LayerActivationFunction.LeakyReLU,// Swish,
               LayerActivationFunction.LeakyReLU,
           }
       );

       Trainer.Settings settings = new Trainer.Settings();
       settings.Population = 100;
       settings.Steps = 1000; 
       settings.ReadyEstimator = (cost, fitness) =>
       {
           return fitness > 0.99f && (cost < 0.2f);
       };
       settings.FitnessEstimator = (network, patterns, labels) =>
            {
                float fittness = 0;
                int c = 0;
                for (int k = 0; k < labels.Length; k++)
                {
                    float[] output = network.FeedForward(patterns[k]);
                    float[] label = labels[k];

                    for (int l = 0; l < output.Length; l++)
                    {
                        float d = label[l] - output[l];
                        fittness += d * d;
                        c++;
                    }
                }
                return 1f - Math.Max(0f, Math.Min(1f, fittness / c));
            };
       
       Trainer.Train
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
                   },
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
                    },
                    new int[]
                    {
                        1, 0
                    },
                    settings
            );
```

# Activation Functions:

```csharp
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU,
        Swish,
        Binary,
        None
```

 