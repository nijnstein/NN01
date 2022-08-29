using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            /// if enabled uses the power of the gpu 
            /// </summary>                           
            public bool GPU { get; set; } = false;

            /// <summary>
            /// a default ready estimation: stop training if fitness or cost reaches some threshold 
            /// </summary>
            public Func<NeuralNetwork, bool> ReadyEstimator { get; set; } = (nn) =>
            {
                return (nn.Cost > 0 && nn.CostDelta < 0.000001) || (nn.Fitness > 0.99f && nn.Cost < 0.005f);
            };

            /// <summary>
            /// default fitness estimator for current state 
            /// </summary>
            public Func<NeuralNetwork, float[][], float[][], float> FitnessEstimator { get; set; } = (network, patterns, labels) =>
            {
                Debug.Assert(labels.Length > 0); 
                if (labels.Length > 0)
                {
                    float fittness = 0;
                    int c = 0;
                    for (int k = 0; k < labels.Length; k++)
                    {
                        float[] output = network.FeedForward(patterns[k]);
                        float[] label = labels[k];

                        fittness += Intrinsics.SumSquaredDifferences(label, output);
                    }
                    return 1f - Math.Max(0f, Math.Min(1f, fittness / (labels.Length * labels[0].Length)));
                }
                return float.NaN; 
            };

            /// <summary>
            /// Default settings 
            /// </summary>
            public static Settings Default = new Settings();
        }
    }
}
