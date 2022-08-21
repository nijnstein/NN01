# NN01
My first neural since computers became fast :) 

NN01 represents the first classifier in my tobe pattern classification library based on machine learning methods. 

# Usage
'''csharp
      
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

''' 
