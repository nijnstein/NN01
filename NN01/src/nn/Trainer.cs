using ILGPU.Runtime;
using NSS;
using NSS.GPU;
using NSS.Neural;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data.SqlTypes;
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
            if (settings.MeanCancelation)
            {
                trainingSamples.CancelMeans();
                testSamples.CancelMeans();
            }

            // prepare gpu (random + memory)  
            IRandom random;
            using GPUContext ctx = GPUContext.Create();
#if DEBUG
            Accelerator acc = settings.GPU ? ctx.CreateCPUAccelerator() : null;
#else
            Accelerator acc = settings.GPU ? ctx.CreateGPUAccelerator() : null;
#endif
            if (settings.GPU && acc != null)
            {
                trainingSamples.AllocateGPU(acc);
                testSamples.AllocateGPU(acc); 
            }
           
            if (settings.Random != null)
            {
                random = settings.Random;
            }
            else
            {
                if(settings.RandomGPU)
                {
                    random = new GPURandom(RandomDistributionInfo.Uniform(0, 1), 128 * 2048, Environment.ProcessorCount * 2, null);
                }
                else
                {
                   random = new CPURandom(RandomDistributionInfo.Default); 
                }
            }
 

            // initialize a population 
            List<NeuralNetwork> networks = new List<NeuralNetwork>(settings.Population);
            networks.Add(network);
            for (int i = 1; i < settings.Population; i++)
            {
                networks.Add(new NeuralNetwork(network, random));
            }


            int istep = 0;
            float currentMomentum = settings.InitialMomentum;
            float currentWeightCost = settings.InitialWeightCost;
            NeuralNetwork best = new NeuralNetwork(network, true);
            
            // shuffle training set 
            trainingSamples.ShuffleIndices(random);

            // timings are nice, but only if used 
            Stopwatch? sw = settings.OnStep != null ? new Stopwatch() : null; 

            for (; istep < settings.Steps; istep++)
            {
                if (sw != null)
                {
                    sw.Reset();
                    sw.Start();
                }
                bool batched = (istep < settings.BatchedStartSteps) || !settings.OnlineTraining;
                int mutationCount = 0;

                // train 1 step
                if (settings.GPU)
                {
                    for (int populationIndex = 0; populationIndex < settings.Population; populationIndex++)
                    {
                        mutationCount += Step(network, trainingSamples, testSamples, settings, random, acc, networks, istep, currentMomentum, batched, populationIndex);
                    }
                }
                else
                {
#if DEBUG
                    for (int j = 0; j < settings.Population; j++)
                    {
                        mutationCount += Step(network, trainingSamples, testSamples, settings, random, acc, networks, istep, currentMomentum, batched, j);
                    }
#else
                    Parallel.For(0, settings.Population, (j) =>
                    {
                        mutationCount = Interlocked.Add(
                            ref mutationCount,
                            Step(network, trainingSamples, testSamples, settings, random, acc, networks, istep, currentMomentum, batched, j));
                    });
#endif
                }

                // shuffle training set 
                trainingSamples.ShuffleIndices(random);

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

                float populationError = networks.Average(x => x.Fitness.Square());


                // notify 
                if (settings.OnStep != null)
                {
                    sw.Stop();
                    settings.OnStep(networks[settings.Population - 1], istep, batched, populationError, (float)sw.Elapsed.TotalMilliseconds, mutationCount);
                }
            }

            // reaching here implies that the last network is the best we could train
            networks[settings.Population - 1].DeepClone(network);

        ready:
           
            if (settings.GPU && acc != null)
            {
                try
                {
                    trainingSamples.ReleaseGPU(acc);
                    testSamples.ReleaseGPU(acc); 
                    acc.Dispose();
                }
                catch(Exception ex)
                {
                    Debug.WriteLine("error disposing accelerator: " + ex.ToString()); 
                }
            }
            if(settings.RandomGPU && settings.Random == null)
            {
                if (random is GPURandom gpuRandom) gpuRandom.Dispose();                 
            }
            return istep;
        }

        private static int Step(NeuralNetwork network, SampleSet trainingSamples, SampleSet testSamples, Settings? settings, IRandom random, Accelerator acc, List<NeuralNetwork> networks, int istep, float currentMomentum, bool batched, int j)
        {
            int mutationCount = 0; 

            // mutate population member if:
            // - not on the first step 
            // - located in bad half 
            if (settings.MutationChance > 0 && istep > 0 && j < settings.Population / 2)
            {
                networks[(int)(j + settings.Population / 2)].DeepClone(networks[j]);
                mutationCount = networks[(int)j].Mutate(random, settings.MutationChance, settings.WeightMutationStrength, settings.BiasMutationStrength);
            }

            // back propagate population
            // 
            if (!batched)
            {
                // setup buffers
                float[][] gamma = new float[network.LayerCount][];
                for (int i = 0; i < network.LayerCount; i++)
                {
                    gamma[i] = new float[network[i].Size];
                }

                // backprop: one sample at a time (online training)
                float cost = 0;
                int sampleCount = Math.Min((int)(settings.MiniBatchSize > 0 ? settings.MiniBatchSize : trainingSamples.SampleCount), trainingSamples.SampleCount);
                for (int k = 0; k < sampleCount; k++)
                {
                    networks[j].BackPropagateOnline(
                        trainingSamples.ShuffledData(k),
                        trainingSamples.ShuffledExpectation(k),
                        gamma,
                        settings.LearningRate,
                        settings.LearningRate,
                        currentMomentum,
                        1E-05f, 1E-07F);

                    trainingSamples.Samples[trainingSamples.ShuffledSample(k).Index].Cost = networks[(int)j].Cost;
                    cost += networks[(int)j].Cost;
                }
                networks[(int)j].Cost = Math.Min(0.999999f, cost / (float)sampleCount * trainingSamples.ClassCount * trainingSamples.Variance);
            }
            else
            {
                // batched training 
                networks[j].BackPropagateBatch(
                    trainingSamples,
                    settings.LearningRate,// * (settings.SoftMax ? 1f : 5f), 
                    settings.LearningRate,// * (settings.SoftMax ? 1f : 5f), 
                    currentMomentum,
                    1E-05f, 1E-07F,
                    null,
                    acc);
            }

            // recalculate fittness 
            networks[(int)j].Fitness = settings.FitnessEstimator(networks[(int)j], trainingSamples) * settings.FitnessEstimator(networks[j], testSamples);
            return mutationCount;
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
            bool softmaxEnabled = settings == null ? Settings.Default.SoftMax : settings.SoftMax;

            return Train(
                network,
                new SampleSet(patterns, classes, network.Output.Size, softmaxEnabled),
                new SampleSet(testPatterns, testClasses, network.Output.Size, softmaxEnabled),
                settings
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, Settings? settings = null)
        {
            return Train(network, ArrayUtils.ConvertTo2D(patterns) , classes, ArrayUtils.ConvertTo2D(testPatterns), testClasses, settings);
        }


    }
}