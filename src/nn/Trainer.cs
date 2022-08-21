using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public static partial class Trainer
    {
        public static int Train(
            NeuralNetwork network,

            float[][] patterns,
            float[][] labels,

            float[][] testPatterns,
            float[][] testLabels,

            Func<NeuralNetwork, float[][], float[][], float> fitnessEstimator,
            Func<float, float, bool> readyEstimator,
            
            Settings settings = null!)
        {
            settings = settings == null ? Settings.Default : settings;
            if (fitnessEstimator == null) fitnessEstimator = settings.FitnessEstimator;
            if (readyEstimator == null) readyEstimator = settings.ReadyEstimator;

            // initialize a mutated population 
            List<NeuralNetwork> networks = new List<NeuralNetwork>(settings.Population);
            networks.Add(network);
            for (int i = 1; i < settings.Population; i++)
            {
                NeuralNetwork clone = network.DeepCopy();
                clone.Mutate(1, settings.MutationStrength);
                networks.Add(clone);
            }

            int istep = 0;
            float currentMomentum = settings.InitialMomentum;
            float currentWeightCost = settings.InitialWeightCost; 

            for (; istep < settings.Steps; istep++)
            {
                // train 1 step
                Parallel.For(0, settings.Population, (j) =>
                {
                    // setup buffers
                    float[][] gamma = new float[network.LayerCount][];
                    for (int i = 0; i < network.LayerCount; i++)
                    {
                        gamma[i] = new float[network[i].Size];
                    }

                    // mutate population if:
                    // - not on the first step 
                    // - located in bad half 
                    if (istep > 0 && j < settings.Population / 2)
                    {
                        networks[j + settings.Population / 2].DeepClone(networks[j]);
                        networks[j].Mutate((int)(1 / settings.MutationChance), settings.MutationStrength);
                    }
                    
                    // back propagate population 
                    for (int k = 0; k < labels.Length; k++)
                    {
                        networks[j].BackPropagate(patterns[k], labels[k], gamma, settings.LearningRate, settings.LearningRate, currentMomentum);
                    }
                    
                    // recalculate fittness 
                    networks[j].Fitness = fitnessEstimator(networks[j], patterns, labels) * fitnessEstimator(networks[j], testPatterns, testLabels);
                });                

                // sort on fitness condition
                networks.Sort();

                // skip ready estimation and mutation if this is the last training step 
                if (istep < settings.Steps - 1)
                {
                    // check if the one of the best n% networks passes the ready test 
                    for (   int j = settings.Population - 1; 
                            j >= (int)Math.Max(settings.Population - (settings.Population * settings.ReadyTestSlice), 0f); 
                            j-- )
                    {
                        if (readyEstimator(networks[j].Cost, networks[j].Fitness))
                        {
                            // clone best mutation into source network 
                            networks[settings.Population - 1].DeepClone(network);
                            goto ready;
                        }
                    }
                }

                // update momemtum 
                currentMomentum = (settings.FinalMomentum > settings.InitialMomentum) ?
                    Math.Min(settings.FinalMomentum, currentMomentum * settings.MomentumChange)
                    :
                    Math.Max(settings.FinalMomentum, currentMomentum * settings.MomentumChange);

                // update momemtum 
                currentWeightCost = (settings.FinalWeightCost > settings.InitialWeightCost) ?
                    Math.Min(settings.FinalWeightCost, currentWeightCost * settings.WeightCostChange)
                    :
                    Math.Max(settings.FinalWeightCost, currentWeightCost * settings.WeightCostChange);
            }

            // reaching here implies that the last network is the best we could train
            networks[settings.Population - 1].DeepClone(network);

        ready:
            return istep;
        }

        #region Train Overloads 
        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, Settings settings = null)
        {
            float[][] CreateOutputLabels(int classCount, int[] classes)
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

            settings = settings == null ? Settings.Default : settings;

            return Train(
                network,
                patterns,
                CreateOutputLabels(network.Output.Size, classes),
                testPatterns,
                CreateOutputLabels(network.Output.Size, testClasses),
                settings.FitnessEstimator,
                settings.ReadyEstimator,
                settings
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, float readyAt, Settings settings = null)
        {
            settings = settings == null ? Settings.Default : settings;

            settings.ReadyEstimator = (cost, fitness) =>
            {
                return fitness > readyAt && 1 - cost > readyAt;
            }; 

            return Train(network, patterns, classes, testPatterns, testClasses, settings);
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float readyAt = 0.9999f, Settings settings = null)
        {
            return Train(network, patterns, classes, patterns, classes, readyAt, settings);
        }

        #endregion 

    }
}
