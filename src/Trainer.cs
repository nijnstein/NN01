using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static class Trainer
    {
        static float[][] CreateOutputLabels(int classCount, int[] classes)
        {
            float[][] labels = new float[classes.Length][];
            for (int i = 0; i < classes.Length; i++)
            {
                labels[i] = new float[classCount];
                for (int j = 0; j < classCount; j++)
                {
                    labels[i][j] = (classes[i] == (j + 1)) ? 1f : 0f;
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

            int steps = 1000,
            int population = 100,
            float mutationChance = 0.01f,
            float mutationStrength = 0.5f)
        {
            // initilize population 
            List<NeuralNetwork> networks = new List<NeuralNetwork>(population);
            networks.Add(network);
            for (int i = 1; i < population; i++)
            {
                NeuralNetwork clone = network.DeepCopy();
                clone.Mutate(1, mutationStrength);
                networks.Add(clone);
            }

            int istep = 0;
            for (; istep < steps; istep++)
            {
                // train 1 step
                for (int j = 0; j < population; j++)
                {
                    for (int k = 0; k < labels.Length; k++)
                    {
                        networks[j].BackPropagate(patterns[k], labels[k]);
                    }
                }

                // estimate fitness 
                for (int j = 0; j < population; j++)
                {
                    networks[j].fitness =
                        fitnessEstimator(networks[j], patterns, labels)
                        *
                        fitnessEstimator(networks[j], testPatterns, testLabels);
                }

                // sort on fitness and stop early past perfect fit 
                networks.Sort();

                if (readyTest(networks[population - 1].cost, networks[population - 1].fitness))
                {
                    break;
                }

                // keep the best half of the population, the worst half is replaced with mutations 
                for (int j = 0; j < population / 2; j++)
                {
                    networks[j + population / 2].DeepClone(networks[j]);
                    networks[j].Mutate((int)(1 / mutationChance), mutationStrength);
                }
            }

            networks[population - 1].DeepClone(network);
            return istep;
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, int steps = 1000, int population = 100, float mutationChance = 0.01f, float mutationStrength = 0.5f, float readyAt = 0.9999f)
        {
            return Train(network, patterns, classes, patterns, classes, steps, population, mutationChance, mutationStrength, readyAt);
        }

        public static int Train(
            NeuralNetwork network,

            float[][] patterns,
            int[] classes,

            float[][] testPatterns,
            int[] testClasses,

            Func<float, float, bool> readyTest,

            int steps = 1000,
            int population = 100,
            float mutationChance = 0.01f,
            float mutationStrength = 0.5f)
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

                steps,
                population,
                mutationChance,
                mutationStrength
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, int steps = 1000, int population = 100, float mutationChance = 0.01f, float mutationStrength = 0.5f, float readyAt = 0.999f)
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
                    return fitness > readyAt && (1 - cost > readyAt);
                },

                steps,
                population,
                mutationChance,
                mutationStrength);

        }
    }
}
