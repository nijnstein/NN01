using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static class Trainer
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
            /// slice of top fit to test for ready 
            /// </summary>
            public float ReadyTestSlice { get; set; } = 0.05f;

            /// <summary>
            /// Default settings 
            /// </summary>
            public static Settings Default = new Settings();
        }

        static float[][] CreateOutputLabels(int classCount, int[] classes)
        {
            float[][] labels = new float[classes.Length][];
            for (int i = 0; i < classes.Length; i++)
            {
                labels[i] = new float[classCount];
                for (int j = 0; j < classCount; j++)
                {
                    labels[i][j] = classes[i] == j + 1 ? 1f : 0f;
                }
            }
            return labels;
        }

        public static int Train(
            NeuralNetwork network,

            float[][] patterns,
            float[][] labels,

            float[][] testPatterns,
            float[][] testLabels,

            Func<NeuralNetwork, float[][], float[][], float> fitnessEstimator,
            Func<float, float, bool> readyTest,

            Settings settings = null)
        {
            settings = settings == null ? Settings.Default : settings;

            // initilize population 
            List<NeuralNetwork> networks = new List<NeuralNetwork>(settings.Population);
            networks.Add(network);
            for (int i = 1; i < settings.Population; i++)
            {
                NeuralNetwork clone = network.DeepCopy();
                clone.Mutate(1, settings.MutationStrength);
                networks.Add(clone);
            }

            int istep = 0;
            for (; istep < settings.Steps; istep++)
            {
                // train 1 step
                Parallel.For(0, settings.Population, (j) =>
                {
                    // back propagate population 
                    for (int k = 0; k < labels.Length; k++)
                    {
                        networks[j].BackPropagate(patterns[k], labels[k], settings.LearningRate);
                    }
                    
                    // recalculate fittness 
                    networks[j].Fitness = fitnessEstimator(networks[j], patterns, labels) * fitnessEstimator(networks[j], testPatterns, testLabels);
                });                

                // sort on fitness condition
                networks.Sort();

                // check if the one of the best n% networks passes the ready test 
                for(int j = settings.Population - 1; j >= (int)Math.Max(settings.Population - (settings.Population * settings.ReadyTestSlice), 0f); j--)
                { 
                    if (readyTest(networks[j].Cost, networks[j].Fitness))
                    {
                        // clone best mutation into source 
                        networks[settings.Population - 1].DeepClone(network);
                        goto ready;
                    }
                }

                // keep the best half of the population, the worst half is replaced with mutations 
                Parallel.For(0, settings.Population / 2, (j) =>
                {
                    networks[j + settings.Population / 2].DeepClone(networks[j]);
                    ///  TODO :  ENSURE AT LEAST 1 NEURON IS MUTATTED which is an issue in small layers 
                    networks[j].Mutate((int)(1 / settings.MutationChance), settings.MutationStrength);
                }); 
            }

            // the last network is the best we could train
            networks[settings.Population - 1].DeepClone(network);

        ready:
            return istep;
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, Settings settings = null, float readyAt = 0.9999f)
        {
            return Train(network, patterns, classes, patterns, classes, settings == null ? Settings.Default : settings, readyAt);
        }

        public static int Train(
            NeuralNetwork network,

            float[][] patterns,
            int[] classes,

            float[][] testPatterns,
            int[] testClasses,

            Func<float, float, bool> readyTest,

            Settings settings = null)
        {
            return Train(
                network,

                // training data 
                patterns,
                CreateOutputLabels(network.Output.Size, classes),

                testPatterns,
                CreateOutputLabels(network.Output.Size, testClasses),

                // estimate fitness using given patterns
                (x, ptrn, lbl) =>
                {
                    float fittness = 0;
                    int c = 0;
                    for (int k = 0; k < lbl.Length; k++)
                    {
                        float[] output = x.FeedForward(ptrn[k]);
                        float[] label = lbl[k];

                        for (int l = 0; l < output.Length; l++)
                        {
                            float d = label[l] - output[l];
                            fittness += d * d;
                            c++;
                        }
                    }
                    return 1f - Math.Max(0f, Math.Min(1f, fittness / c));
                },
                readyTest,
                settings == null ? Settings.Default : settings
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, Settings settings = null, float readyAt = 0.999f)
        {
            return Train(
                network,

                // training data 
                patterns,
                classes,

                testPatterns,
                testClasses,

                // test if training is ready before completing all steps 
                (cost, fitness) =>
                {
                    return fitness > readyAt && 1 - cost > readyAt;
                },

                settings == null ? Settings.Default : settings);

        }
    }
}
