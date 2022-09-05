using ILGPU.Runtime;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    public partial class Trainer
    {
        public GPUContext? GPUContext;

        public NeuralNetwork? Network;
        public List<NeuralNetwork>? Networks;
        public Settings? settings;

        private int istep;
        private float currentMomentum;
        private float currentWeightCost;
     

        public static int Train(
            NeuralNetwork network,
            SampleSet trainingSamples, 
            SampleSet testSamples,
            Settings? settings = null)
        {
            Debug.Assert(network != null);
            Debug.Assert(trainingSamples != null);
            Debug.Assert(testSamples != null); 

            settings = settings == null ? Settings.Default : settings;
 
            // prepare samples
            if(settings.MeanCancelation)
            {
                trainingSamples.CancelMeans();
                testSamples.CancelMeans();                                 
            }

            // initialize a population 
            List<NeuralNetwork> networks = new List<NeuralNetwork>(settings.Population);
            networks.Add(network);
            for (int i = 1; i < settings.Population; i++)
            {
                networks.Add(new NeuralNetwork(network, false));
            }

            int istep = 0;
            float currentMomentum = settings.InitialMomentum;
            float currentWeightCost = settings.InitialWeightCost;
            NeuralNetwork best = new NeuralNetwork(network);
            
            // shuffle training set 
            trainingSamples.ShuffleIndices(Random.Shared);


            for (; istep < settings.Steps; istep++)
            {
                // train 1 step
#if DEBUG
                for (int j = 0; j < settings.Population; j++)
                {
#else
                Parallel.For(0, settings.Population, (j) =>
                {
#endif
                    // setup buffers
                    float[][] gamma = new float[network.LayerCount][];
                    for (int i = 0; i < network.LayerCount; i++)
                    {
                        gamma[i] = new float[network[i].Size];
                    }

                    // mutate population member if:
                    // - not on the first step 
                    // - located in bad half 
                    if (settings.MutationChance > 0 && istep > 0 && j < settings.Population / 2)
                    {
                        networks[(int)(j + settings.Population / 2)].DeepClone(networks[j]);
                        networks[(int)j].Mutate(settings.MutationChance, settings.WeightMutationStrength, settings.BiasMutationStrength);
                    }

                    // back propagate population
                    // 
                    //   should the cost not be averge of all samples
                    //  
                    float cost = 0;
                    int sampleCount = Math.Min((int)(settings.MiniBatchSize > 0 ? settings.MiniBatchSize : trainingSamples.SampleCount), trainingSamples.SampleCount);


                    if (settings.OneByOne)
                    {
                        for (int k = 0; k < sampleCount; k++)
                        {
                            networks[j].BackPropagate(
                                trainingSamples.ShuffledData(k),
                                trainingSamples.ShuffledExpectation(k),
                                gamma, settings.LearningRate, settings.LearningRate, currentMomentum, 1E-05f, 1E-07F);

                            trainingSamples.Samples[k].Cost = networks[(int)j].Cost;
                            cost += networks[(int)j].Cost;
                        }
                    }
                    else
                    {
                        networks[j].BackPropagate(trainingSamples, gamma, settings.LearningRate, settings.LearningRate, currentMomentum, 1E-05f, 1E-07F);
                    }


                    networks[(int)j].Cost = Math.Min(0.999999f, cost / (float)sampleCount * trainingSamples.ClassCount);

                    // recalculate fittness 
                    networks[(int)j].Fitness = settings.FitnessEstimator(networks[(int)j], trainingSamples) * settings.FitnessEstimator(networks[j], testSamples);
#if DEBUG
                }
#else
                });
#endif

                // shuffle training set 
                trainingSamples.ShuffleIndices(Random.Shared);

                // sort the training set
                // trainingSamples.SortOnErrorAlternatingOnClass(); 

                // sort on fitness condition
                networks.Sort();

                // skip ready estimation and mutation if this is the last training step 
                if (istep < settings.Steps - 1)
                {
                    // check if the one of the best n% networks passes the ready test 
                    if (EstimateIfReady(network, settings.ReadyEstimator, settings, networks))
                    {
                        goto ready;
                    }
                }

                // update momemtum 
                currentMomentum = (settings.FinalMomentum > settings.InitialMomentum) ?
                    Math.Min(settings.FinalMomentum, currentMomentum * settings.MomentumChange)
                    :
                    Math.Max(settings.FinalMomentum, currentMomentum * settings.MomentumChange);

                // update weight resistance 
                currentWeightCost = (settings.FinalWeightCost > settings.InitialWeightCost) ?
                    Math.Min(settings.FinalWeightCost, currentWeightCost * settings.WeightCostChange)
                    :
                    Math.Max(settings.FinalWeightCost, currentWeightCost * settings.WeightCostChange);


                // keep best network around
                if (best.Fitness < networks[settings.Population - 1].Fitness)
                {
                    networks[settings.Population - 1].DeepClone(best);
                }
                else
                {
                    if (best.Fitness > networks[settings.Population - 1].Fitness)
                    {
                        best.DeepClone(networks[0]);
                        networks.Sort();
                    }
                }

                // notify 
                if (settings.OnStep != null)
                {
                    settings.OnStep(networks[settings.Population - 1], istep);
                }
            }

            // reaching here implies that the last network is the best we could train
            networks[settings.Population - 1].DeepClone(network);

        ready:
            return istep;
        }

        public static bool EstimateIfReady(NeuralNetwork network, Func<NeuralNetwork, bool> readyEstimator, Settings settings, List<NeuralNetwork> networks)
        {
            for (int j = settings.Population - 1;
                    j >= (int)Math.Max(settings.Population - (settings.Population * settings.ReadyTestSlice), 0f);
                    j--)
            {
                if (readyEstimator(networks[j]))
                {
                    // clone best mutation into source network 
                    networks[j].DeepClone(network);
                    return true;
                }
            }

            return false;
        }

        public static int Train(NeuralNetwork network, float[,] patterns, int[] classes, float[,] testPatterns, int[] testClasses, Settings? settings = null)
        {
            return Train(
                network,
                new SampleSet(patterns, classes, network.Output.Size),
                new SampleSet(testPatterns, testClasses, network.Output.Size),
                settings
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, Settings? settings = null)
        {
            return Train(network, ArrayUtils.ConvertTo2D(patterns) , classes, ArrayUtils.ConvertTo2D(testPatterns), testClasses, settings);
        }


    }
}