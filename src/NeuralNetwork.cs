using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{


    public class NeuralNetwork : IComparable
    {
        private Layer[] Layers; 

        public float Fitness = 0;
        public float LearningRate = 0.01f;
        public float Cost = 0;

        public int LayerCount => Layers == null ? 0 : Layers.Length;
        public Layer Input => Layers[0];
        public Layer Output => Layers[Layers.Length - 1];

        public NeuralNetwork(int[] layerSizes, LayerActivationFunction[] activations)
        {
            this.Layers = new Layer[layerSizes.Length];

            Layers[0] = new InputLayer(layerSizes[0]);

            for (int i = 1; i < Layers.Length; i++)
            {
                switch (activations[i - 1])
                {
                    case LayerActivationFunction.ReLU:
                        Layers[i] = new ReLuLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.LeakyReLU:
                        Layers[i] = new LeakyReLuLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Tanh:
                        Layers[i] = new TanhLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Sigmoid:
                        Layers[i] = new SigmoidLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.SoftMax:
                        Layers[i] = new SoftMaxLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Swish:
                        Layers[i] = new SwishLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Linear:
                        // reduces model to linear regression, no backprop 
                        Layers[i] = new LinearLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                   case LayerActivationFunction.Binary:
                        // no backprop 
                        Layers[i] = new BinaryLayer(layerSizes[i], layerSizes[i - 1]);
                        break;
                }
            }
        }
        public NeuralNetwork(NeuralNetwork other)
        {
            this.Layers = new Layer[other.Layers.Length];

            Layers[0] = new InputLayer(other.Layers[0].Size);

            for (int i = 1; i < other.Layers.Length; i++)
            {
                switch (other.Layers[i].ActivationType)
                {
                    case LayerActivationFunction.ReLU:
                        Layers[i] = new ReLuLayer(other.Layers[i].Size, other.Layers[i - 1].Size); 
                        break;

                    case LayerActivationFunction.LeakyReLU:
                        Layers[i] = new LeakyReLuLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Tanh:
                        Layers[i] = new TanhLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Sigmoid:
                        Layers[i] = new SigmoidLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.SoftMax:
                        Layers[i] = new SoftMaxLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Swish:
                        Layers[i] = new SwishLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Linear:
                        // reduces model to linear regression, no backprop 
                        Layers[i] = new LinearLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Binary:
                        // no backprop 
                        Layers[i] = new BinaryLayer(other.Layers[i].Size, other.Layers[i - 1].Size);
                        break;
                }
            }

            other.DeepClone(this);
        }


        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");
            for (int i = 0; i < Layers.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append($"-{Layers[i].ActivationType}-");
                }
                sb.Append($"{Layers[i].Size}");
            }
            sb.Append("]");
            return sb.ToString(); 
        }

     
        /// <summary>
        /// feed forward, inputs -> outputs.
        /// </summary>
        public float[] FeedForward(float[] inputs)
        {   
            // set input 
            for (int i = 0; i < inputs.Length; i++)
            {
                Layers[0].Neurons[i] = inputs[i];
            }

            // propagate state through layers 
            for (int i = 1; i < Layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                Layers[i].Activate(Layers[i - 1]);
            }

            // return the output neuron state 
            return Layers[Layers.Length - 1].Neurons;
        }
    


        public void BackPropagate(float[] inputs, float[] expected)
        {
            // ensure proper neuron populations
            float[] output = FeedForward(inputs);

            Cost = 0;
            float[] delta = new float[output.Length]; 
            for (int i = 0; i < delta.Length; i++)
            {
                // precalculate delta 
                delta[i] = output[i] - expected[i]; 

                // calculate cost of network 
                Cost += (float)Math.Pow(delta[i], 2);
            }

            // setup 'gamma'
            float[][] gamma = new float[Layers.Length][];
            for (int i = 0; i < Layers.Length; i++)
            {
                gamma[i] = new float[Layers[i].Size]; 
            }

            // calculate 'gamma' for the last layer
            Layers[Layers.Length - 1].CalculateGamma(delta, gamma[Layers.Length - 1], output); 

            // calculate new weights and biases for the last layer in the network 
            for (int i = 0; i < Layers[Layers.Length - 1].Size; i++)
            {
                Layers[Layers.Length - 2].Biases[i] -= gamma[Layers.Length - 1][i] * LearningRate;
                
                // apply some learning... move in direction of result using gamma 
                for (int j = 0; j < Layers[Layers.Length - 2].Size; j++)
                {
                    Layers[Layers.Length - 1].Weights[i][j] 
                        -= 
                        gamma[Layers.Length - 1][i]
                        *
                        Layers[Layers.Length - 2].Neurons[j] 
                        * LearningRate; 
                }
            }

            // now propagate backward 
            for (int i = Layers.Length - 2; i > 0; i--)
            {
                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < Layers[i].Size; j++)
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * Layers[i + 1].Weights[k][j];
                    }
                    
                    Layers[i].CalculateGamma(gamma[i]); 
                }

                // modify bias to outputlayer 
                for (int j = 0; j < Layers[i].Size; j++)
                {
                    // update bias 
                    Layers[i].Biases[j] -= gamma[i][j] * LearningRate;

                    // modify weight to this layers inputs
                    for (int k = 0; k < Layers[i - 1].Size; k++)
                    {
                        Layers[i].Weights[j][k] -= gamma[i][j] * Layers[i - 1].Neurons[k] * LearningRate;
                    }
                }
            }
        }


        public static float RandomRange(float low, float high) => Random.Shared.NextSingle() * (high - low) + low;
       
        //Genetic implementations down onwards until save.

        public void Mutate(int high, float val) //used as a simple mutation function for any genetic implementations.
        {
            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < Layers[i].Biases.Length; j++)
                {                           
                    Layers[i].Biases[j] = ((Random.Shared.NextSingle() * high) <= 2) 
                        ?
                        Layers[i].Biases[j] += RandomRange(-val, val)
                        :
                        Layers[i].Biases[j];
                }
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < Layers[i].Weights.Length; j++)
                {
                    for (int k = 0; k < Layers[i].Weights[j].Length; k++)
                    {
                        Layers[i].Weights[j][k] = (RandomRange(0f, high) <= 2) ? Layers[i].Weights[j][k] += RandomRange(-val, val) : Layers[i].Weights[j][k];
                    }
                }
            }
        }

        public void DeepClone(NeuralNetwork into)
        {
            into.Cost = Cost;
            into.LearningRate = LearningRate;
            into.Fitness = Fitness;

            for (int i = 0; i < Layers.Length; i++)
            {
                for(int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    into.Layers[i].Neurons[j] = Layers[i].Neurons[j]; 
                }

                if (!Layers[i].IsInput)
                {
                    for (int j = 0; j < Layers[i].Biases.Length; j++)
                    {
                        into.Layers[i].Biases[j] = Layers[i].Biases[j];
                    }
                    for (int j = 0; j < Layers[i].Weights.Length; j++)
                    {
                        for (int k = 0; k < Layers[i].Weights[j].Length; k++)
                        {
                            into.Layers[i].Weights[j][k] = Layers[i].Weights[j][k]; 
                        }
                    }
                }
            }
        }

        public NeuralNetwork DeepCopy()
        {
            return new NeuralNetwork(this);            
        }
 
        public int CompareTo(object? obj)
        {
            return CompareTo((obj as NeuralNetwork)!);
        }

        public int CompareTo(NeuralNetwork other)
        {
            if (other == null) return 1;

            if (Fitness > other.Fitness)
                return 1;
            else if (Fitness < other.Fitness)
                return -1;
            else
                return 0;
        }

    }


}

