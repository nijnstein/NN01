using NSS;
using NSS.Neural;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
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
            public int Population { get; set; } = 32;

            /// <summary>
            /// learning rate
            /// </summary>
            public float LearningRate = 0.03f;
            public float BatchedLearningRate = 0.15f;

            /// <summary>
            /// chance of a single neurons weight to change
            /// - this is not a percentage and heavily depends on the random distribution used, if the random is higher it mutates
            /// </summary>
            public float MutationChance { get; set; } = .3f;

            public float MaximalMutationChance { get; set; } = .85f;
            public float MinimalMutationChance { get; set; } = .05f;

            /// <summary>
            /// strenght of any mutation if it occurs 
            /// </summary>
            public float BiasMutationStrength { get; set; } = 0.06f;

            /// <summary>
            /// strenght of any mutation if it occurs 
            /// </summary>
            public float WeightMutationStrength { get; set; } = 0.2f;

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
            /// bring mean of each sample close 2 zero
            /// </summary>
            public bool MeanCancelation { get; set; } = false;

            public int MiniBatchSize { get; set; } = 0;

            public int BatchedStartSteps { get; set; } = 0;

            /// <summary>
            /// - if true, performs online training (sample after sample
            /// - if false, perform batch training 
            /// </summary>
            public bool OnlineTraining { get; set; } = true;

            public bool SoftMax { get; set; } = false;

            /// <summary>
            /// weight dropout factor 
            /// </summary>
            public float DropOutFactor { get; set; } = 0.01f; 

            /// <summary>
            /// if enabled uses the power of the gpu 
            /// </summary>                           
            public bool GPU { get; set; } = false;

            public bool RandomGPU { get; set; } = true;

            public IRandom? Random { get; set; }

            /// <summary>
            /// a default ready estimation: stop training if fitness or cost reaches some threshold 
            /// </summary>
            public Func<NeuralNetwork, bool> ReadyEstimator { get; set; } = (nn) =>
            {
                return (nn.Cost > 0 && nn.CostDelta < 0.0000001) || (nn.Fitness > 0.999999f && nn.Cost < 0.000001f);
            };

            /// <summary>
            /// default fitness estimator for current state 
            /// </summary>
            public Func<NeuralNetwork, SampleSet, float> FitnessEstimator { get; set; } = (network, samples) =>
            {
                Debug.Assert(samples.SampleCount > 0); 
                if (samples.SampleCount > 0)
                {
                    float fittness = 0;
                    int c = 0;
                    object locker = new object();
                    for (int k = 0; k < samples.SampleCount; k++)
                    {
                        Span<float> output = network.FeedForward(samples.SampleData(k));
                        Span<float> label = samples.SampleExpectation(k);

                        float f = Intrinsics.SumSquaredDifferences(label, output);
                        lock (locker)
                        {
                            fittness += f;
                        }
                    }
                    return 1f - Math.Max(0f, Math.Min(1f, fittness / (samples.SampleCount * samples.ClassCount)));
                }
                return float.NaN; 
            };

            public delegate void OnStepEvent(NeuralNetwork network, int step, bool batched, float populationError, float totalMilliseconds, int mutationCount); 
            public OnStepEvent OnStep = null;


            /// <summary>
            /// Default settings 
            /// </summary>
            public static Settings Default = new Settings();
        }
    }
}
