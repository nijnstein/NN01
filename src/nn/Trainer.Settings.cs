using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public partial class Trainer
    {
        public class Settings
        {
            /// <summary>
            /// training steps 
            /// </summary>
            public int Steps { get; set; } = 1000;

            /// <summary>
            /// training population 
            /// </summary>
            public int Population { get; set; } = 100;

            /// <summary>
            /// learning rate
            /// </summary>
            public float LearningRate = 0.01f;

            /// <summary>
            /// chance of a single neurons weight to change
            /// </summary>
            public float MutationChance { get; set; } = 0.01f;

            /// <summary>
            /// strenght of any mutation if it occurs 
            /// </summary>
            public float MutationStrength { get; set; } = 0.5f;

            /// <summary>
            /// initial cost of weight changes
            /// </summary>
            public float InitialWeightCost = 0.00001f; 

            /// <summary>
            /// final cost of weight changes after n steps of applying change factor
            /// </summary>
            public float FinalWeightCost = 0.001f;

            /// <summary>
            /// rate of change for weightcost 
            /// </summary>
            public float WeightCostChange = 1.01f;

            /// <summary>
            /// initial momentum moving weights and biases, 
            /// </summary>
            public float InitialMomentum = 1.5f;

            /// <summary>
            /// momentum change on each step 
            /// </summary>
            public float MomentumChange = 0.95f; 

            /// <summary>
            /// final momentum after n steps of applying loss
            /// </summary>
            public float FinalMomentum = 0.5f; 

            /// <summary>
            /// slice of top fit to test for ready 
            /// </summary>
            public float ReadyTestSlice { get; set; } = 0.05f;

            /// <summary>
            /// a default ready estimation: stop training if fitness or cost reaches some threshold 
            /// </summary>
            public Func<float, float, bool> ReadyEstimator { get; set; } = (cost, fitness) =>
            {
                return fitness > 0.999f && (cost < 0.005f);
            };

            /// <summary>
            /// default fitness estimator for current state 
            /// </summary>
            public Func<NeuralNetwork, float[][], float[][], float> FitnessEstimator { get; set; } = (network, patterns, labels) =>
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

            /// <summary>
            /// Default settings 
            /// </summary>
            public static Settings Default = new Settings();
        }
    }
}
