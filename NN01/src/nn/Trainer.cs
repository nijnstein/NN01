using ILGPU.Runtime;
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

        public void Reset(NeuralNetwork network, IEnumerable<NeuralNetwork> others, Settings settings = null!)
        {
            this.Network = network;
            this.settings = settings == null ? Settings.Default : settings;

            // initialize a population 
            Networks = new List<NeuralNetwork>(this.settings.Population);
            Networks.Add(network);
            if (others != null)
            {
                Networks.AddRange(others);
            }

            while(Networks.Count < this.settings.Population)
            {
                Networks.Add(new NeuralNetwork(network, false));
            }

            istep = 0;
            currentMomentum = this.settings.InitialMomentum;
            currentWeightCost = this.settings.InitialWeightCost;
        }

        public void Reset(NeuralNetwork network, Settings settings = null!)
        {
            Reset(network, null, settings); 
        }

        public void Step(
            float[][] patterns,
            float[][] labels,

            float[][] testPatterns,
            float[][] testLabels,

            Func<NeuralNetwork, float[][], float[][], float> fitnessEstimator = null!)
        {
            Step(
                patterns, MathEx.Variance(patterns, MathEx.Average(patterns)), labels,
                testPatterns, testLabels, 
                fitnessEstimator);
        }

        public void Step(
            float[][] patterns, float variance,
            float[][] labels,

            float[][] testPatterns,
            float[][] testLabels,

            Func<NeuralNetwork, float[][], float[][], float> fitnessEstimator = null!)
        {
            if (settings == null || Network == null || Networks == null || patterns == null || labels == null)
            {
                throw new ArgumentNullException();
            }
            if (fitnessEstimator == null) fitnessEstimator = settings!.FitnessEstimator;
            if (testPatterns == null)
            {
                testPatterns = patterns;
                testLabels = labels;
            }

     
                // train 1 step
                Parallel.For(0, settings.Population, (j) =>
                {
                    Accelerator? acc = GPUContext != null ? GPUContext.CreateAccelerator() : null;

                    // init gamma buffer foreach network if needed 
                    if (Networks[j].Gamma == null)
                    {
                        float[][] gamma = new float[Networks[j].LayerCount][];
                        for (int i = 0; i < Networks[j].LayerCount; i++)
                        {
                            gamma[i] = new float[Networks[j][i].Size];
                        }
                        Networks[j].Gamma = gamma;
                    }

                    // mutate population if:
                    // - not on the first step 
                    // - located in bad half 
                    if (istep > 0 && j < settings.Population / 2)
                    {
                        Networks[j + settings.Population / 2].DeepClone(Networks[j]);
                        Networks[j].Mutate((int)(1 / settings.MutationChance), settings.MutationStrength);
                    }

                    // back propagate population 
                    for (int k = 0; k < labels.Length; k++)
                    {
                        Networks[j].BackPropagate(patterns[k], labels[k], Networks[j].Gamma, settings.LearningRate, settings.LearningRate, currentMomentum);
                    }

                    // recalculate fittness 
                    Networks[j].Fitness = fitnessEstimator(Networks[j], patterns, labels) * fitnessEstimator(Networks[j], testPatterns, testLabels);

                    if (acc != null) acc.Dispose();
                });
            

            // sort on fitness condition
            Networks.Sort();

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

            // the last network in the list is the best
            Networks[settings.Population - 1].DeepClone(Network);
        }

        public bool EstimateIfReady(Func<NeuralNetwork, bool>? readyEstimator = null)
        {
            Debug.Assert(settings != null);
            Debug.Assert(Network != null);
            if (readyEstimator == null)
            {
                readyEstimator = settings.ReadyEstimator; 
            }
            return Trainer.EstimateIfReady(Network!, readyEstimator, settings!, Networks!);
        }


        #region Older static code, still used in first tests.. todo: refactor that away 
        public static int Train(
            NeuralNetwork network,

            float[][] patterns,
            float[][] labels,

            float[][] testPatterns,
            float[][] testLabels,

            Func<NeuralNetwork, float[][], float[][], float> fitnessEstimator,
            Func<NeuralNetwork, bool> readyEstimator,
            
            Settings settings = null!)
        {
            settings = settings == null ? Settings.Default : settings;
            if (fitnessEstimator == null) fitnessEstimator = settings.FitnessEstimator;
            if (readyEstimator == null) readyEstimator = settings.ReadyEstimator;

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
                        networks[j].BackPropagate(patterns[k], labels[k], gamma, settings.LearningRate, settings.LearningRate, currentMomentum, 1E-05f, 1E-07F);
                    }

                    // recalculate fittness 
                    networks[j].Fitness = fitnessEstimator(networks[j], patterns, labels) * fitnessEstimator(networks[j], testPatterns, testLabels);
#if DEBUG
                }
#else
                });                
#endif 

                // sort on fitness condition
                networks.Sort();

                // skip ready estimation and mutation if this is the last training step 
                if (istep < settings.Steps - 1)
                {
                    // check if the one of the best n% networks passes the ready test 
                    if(EstimateIfReady(network, readyEstimator, settings, networks))
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

            settings.ReadyEstimator = (nn) =>
            {
                return nn.CostDelta < 0.0001 || (nn.Fitness > readyAt && nn.Cost < nn.Cost - readyAt);
            };

            return Train(network, patterns, classes, testPatterns, testClasses, settings);
        }

    
        #endregion

    }
}
