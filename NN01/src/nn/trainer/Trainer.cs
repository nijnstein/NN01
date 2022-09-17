using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
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
        public class TrainerInfo
        {
            public int step;
            public float weightLearningRate;
            public float biasLearningRate;
            public float currentMomentum;
            public float currentWeightCost;

            // if drop out is enabled, then each network uses these bitmaps to dropout weights 
            public BitBuffer2D[] dropOutLayers; 

            public SampleSet trainingSet;
            public SampleSet testSet;
            public Settings settings;

            public Accelerator accelerator;
            public GpuNetworkDataHost gpuDataHost; 
        }

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
            Stopwatch? sw = settings.OnStep != null ? new Stopwatch() : null;
            if (sw != null) sw.Start(); 

            // prepare samples
            if (settings.MeanCancelation)
            {
                trainingSamples.CancelMeans();
                testSamples.CancelMeans();
            }

            // random 
            if (settings.Random == null)
            {
                settings.Random = new CPURandom(RandomDistributionInfo.Uniform());
            }
            
            List<NeuralNetwork> networks = new List<NeuralNetwork>(settings.Population);
            networks.Add(network);
            for (int i = 1; i < settings.Population; i++)
            {
                networks.Add(new NeuralNetwork(network, settings.Random));
            }

            GPUContext gpuContext = settings.GPU ? GPUContext.Create() : null;
            TrainerInfo info = new TrainerInfo()
            {
                settings = settings,
                step = 0,
                currentMomentum = settings.InitialMomentum,
                currentWeightCost = settings.InitialWeightCost,
                weightLearningRate = settings.LearningRate,
                biasLearningRate = settings.LearningRate,
                testSet = testSamples,
                trainingSet = trainingSamples,
                accelerator = gpuContext != null ? gpuContext.CreateGPUAccelerator() : null
            };
            if (info.accelerator != null)
            {
                trainingSamples.AllocateAndCopyToGPU(info.accelerator, info.settings.Population); 
                testSamples.AllocateAndCopyToGPU(info.accelerator, info.settings.Population);
                info.gpuDataHost = GpuNetworkDataHost.AllocatePopulation(info.accelerator, network, info.settings.Population); 
            }

            NeuralNetwork best = new NeuralNetwork(network, true);
            best.Fitness = 0;
            best.Cost = float.MaxValue;
            
            // shuffle training set 
            trainingSamples.ShuffleIndices(settings.Random);

            // timings are nice, but only if used 
            if(sw != null && settings.OnStep != null)
            {
                sw.Stop();
                settings.OnStep(network, -1, false, 0, (float)sw.Elapsed.TotalMilliseconds, 0);
            }

            for (; info.step < settings.Steps; info.step++)
            {
                if (sw != null)
                {
                    sw.Reset();
                    sw.Start();
                }
                bool batched = /*istep > 0 &&*/ ((info.step < settings.BatchedStartSteps) || !settings.OnlineTraining);
                int mutationCount = 0;


                // train 1 step
#if DEBUG
                    for (int j = 0; j < settings.Population; j++)
                    {
                         mutationCount += Step(network, info, settings.Random, networks, batched, j);
                    }
#else
                    Parallel.For(0, settings.Population, (j) =>
                    {
                        mutationCount = Interlocked.Add(
                            ref mutationCount,
                            Step(network, info, settings.Random, networks, batched, j));
                    });
#endif
                                    

                // shuffle training set - no use if batched 
                if (!batched)
                {
                    trainingSamples.ShuffleIndices(settings.Random);
                }

                // sort the training set
                // trainingSamples.SortOnErrorAlternatingOnClass(); 

                // sort on fitness condition
                networks.Sort();

                // skip ready estimation and mutation if this is the last training step 
                if (info.step < settings.Steps - 1)
                {
                    // check if the one of the best n% networks passes the ready test 
                    if (EstimateIfReady(network, settings.ReadyEstimator, settings, networks))
                    {
                        goto ready;
                    }
                }

                // update momemtum 
                info.currentMomentum = (settings.FinalMomentum > settings.InitialMomentum) ?
                    Math.Min(settings.FinalMomentum, info.currentMomentum * settings.MomentumChange)
                    :
                    Math.Max(settings.FinalMomentum, info.currentMomentum * settings.MomentumChange);

                // update weight resistance 
                info.currentWeightCost = (settings.FinalWeightCost > settings.InitialWeightCost) ?
                    Math.Min(settings.FinalWeightCost, info.currentWeightCost * settings.WeightCostChange)
                    :
                    Math.Max(settings.FinalWeightCost, info.currentWeightCost * settings.WeightCostChange);


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

                float populationError = networks.Average(x => x.Fitness);//.Square());
           
                // notify 
                if (settings.OnStep != null)
                {
                    sw.Stop();
                    settings.OnStep(networks[settings.Population - 1], info.step, batched, populationError, (float)sw.Elapsed.TotalMilliseconds, mutationCount);
                }
            }

            // reaching here implies that the last network is the best we could train
            networks[settings.Population - 1].DeepClone(network);

        ready:
            if (settings.GPU)
            {
                info.gpuDataHost.ReleaseAllocation(); 
                info.trainingSet.ReleaseGPU(info.accelerator);
                info.testSet.ReleaseGPU(info.accelerator); 
                info.accelerator.Dispose();
                gpuContext.Dispose();
            }
            return info.step;
        }

        private static int Step(NeuralNetwork network, TrainerInfo info, IRandom random, List<NeuralNetwork> networks, bool batched, int j)
        {
            int mutationCount = 0;

            // randomize any dropout layer 
            for (int i = 1; i < networks[j].LayerCount; i++)
            {
                if (networks[j][i].ActivationType == LayerType.Dropout)
                {
                    Dropout layer = networks[j][i] as Dropout; 
                    if(layer != null)
                    {
                        layer.RandomizeDropout(random); 
                    }
                }
            }

            // back propagate population
            // 
            if (!batched)
            {
                // mutate population member if:
                // - not on the first step 
                // - located in bad half 
                mutationCount = MutatePopulationMember(info.settings, random, networks, info.step, j);

                // setup buffers
                float[][] gamma = new float[network.LayerCount][];
                for (int i = 0; i < network.LayerCount; i++)
                {
                    gamma[i] = new float[network[i].Size];
                }

                // backprop: one sample at a time (online training)
                float cost = 0;
                int sampleCount = Math.Min((int)(info.settings.MiniBatchSize > 0 ? info.settings.MiniBatchSize : info.trainingSet.SampleCount), info.trainingSet.SampleCount);
                for (int k = 0; k < sampleCount; k++)
                {
                    networks[j].BackPropagateOnline(
                        info.trainingSet,
                        k,
                        gamma,
                        info.settings.LearningRate,
                        info.settings.LearningRate,
                        info.currentMomentum,
                        1E-05f, 1E-07F);

                    info.trainingSet.Samples[info.trainingSet.ShuffledSample(k).Index].Cost = networks[(int)j].Cost;
                    cost += networks[(int)j].Cost;
                }
                networks[(int)j].Cost = Math.Min(
                    0.999999f, 
                    cost / (float)sampleCount * info.trainingSet.ClassCount * info.trainingSet.Variance);
            }
            else
            {
                mutationCount = MutatePopulationMember(info.settings, random, networks, info.step, j);

                // batched training 
                // networks[j].
                networks[j].BackPropagateBatchCPU(
                    j,
                    info.trainingSet,
                    info.settings.BatchedLearningRate,// * (settings.SoftMax ? 1f : 5f), 
                    info.settings.BatchedLearningRate,// * (settings.SoftMax ? 1f : 5f), 
                    info.currentMomentum,
                    1E-05f, 1E-07F,
                    random);
            }

            networks[(int)j].Fitness =
                info.settings.FitnessEstimator(networks[(int)j], info.trainingSet)
                *
                info.settings.FitnessEstimator(networks[j], info.testSet);

            return mutationCount;
        }

        private static int MutatePopulationMember(Settings? settings, IRandom random, List<NeuralNetwork> networks, int istep, int populationIndex)
        {
            if (settings.MutationChance > 0 && istep > 0 && populationIndex < settings.Population / 2)
            {
                // clone a better netwok into this one
                networks[(int)(populationIndex + settings.Population / 2)].DeepClone(networks[populationIndex]);

                // and mutate it according to settings 
                return networks[(int)populationIndex].Mutate(
                    random, 
                    settings.MutationChance, 
                    settings.WeightMutationStrength, 
                    settings.BiasMutationStrength);
            }

            return 0;
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
                new SampleSet(patterns, classes, network.Output.Size, softmaxEnabled, 1),
                new SampleSet(testPatterns, testClasses, network.Output.Size, softmaxEnabled, 1),
                settings
            );
        }

        public static int Train(NeuralNetwork network, float[][] patterns, int[] classes, float[][] testPatterns, int[] testClasses, Settings? settings = null)
        {
            return Train(network, ArrayUtils.ConvertTo2D(patterns) , classes, ArrayUtils.ConvertTo2D(testPatterns), testClasses, settings);
        }


    }
}