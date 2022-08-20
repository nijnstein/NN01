using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{


    public class NeuralNetwork : IComparable
    {
        private Layer[] layers; 

        public float fitness = 0;
        public float learningRate = 0.01f;
        public float cost = 0;

        public int LayerCount => layers == null ? 0 : layers.Length;
        public Layer Input => layers[0];
        public Layer Output => layers[layers.Length - 1];

        public NeuralNetwork(int[] layerSizes, LayerActivationFunction[] activations)
        {
            this.layers = new Layer[layerSizes.Length];

            layers[0] = new InputLayer(layerSizes[0]);

            for (int i = 1; i < layers.Length; i++)
            {
                switch (activations[i - 1])
                {
                    case LayerActivationFunction.ReLU:
                        layers[i] = new ReLuLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.LeakyReLU:
                        layers[i] = new LeakyReLuLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Tanh:
                        layers[i] = new TanhLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Sigmoid:
                        layers[i] = new SigmoidLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.SoftMax:
                        layers[i] = new SoftMaxLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Swish:
                        layers[i] = new SwishLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                    case LayerActivationFunction.Linear:
                        // reduces model to linear regression, no backprop 
                        layers[i] = new LinearLayer(layerSizes[i], layerSizes[i - 1]);
                        break;

                   case LayerActivationFunction.Binary:
                        // no backprop 
                        layers[i] = new BinaryLayer(layerSizes[i], layerSizes[i - 1]);
                        break;
                }
            }
        }
        public NeuralNetwork(NeuralNetwork other)
        {
            this.layers = new Layer[other.layers.Length];

            layers[0] = new InputLayer(other.layers[0].Size);

            for (int i = 1; i < other.layers.Length; i++)
            {
                switch (other.layers[i].ActivationType)
                {
                    case LayerActivationFunction.ReLU:
                        layers[i] = new ReLuLayer(other.layers[i].Size, other.layers[i - 1].Size); 
                        break;

                    case LayerActivationFunction.LeakyReLU:
                        layers[i] = new LeakyReLuLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Tanh:
                        layers[i] = new TanhLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Sigmoid:
                        layers[i] = new SigmoidLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.SoftMax:
                        layers[i] = new SoftMaxLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Swish:
                        layers[i] = new SwishLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Linear:
                        // reduces model to linear regression, no backprop 
                        layers[i] = new LinearLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;

                    case LayerActivationFunction.Binary:
                        // no backprop 
                        layers[i] = new BinaryLayer(other.layers[i].Size, other.layers[i - 1].Size);
                        break;
                }
            }

            other.DeepClone(this);
        }


        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");
            for (int i = 0; i < layers.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append($"-{layers[i].ActivationType}-");
                }
                sb.Append($"{layers[i].Size}");
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
                layers[0].neurons[i] = inputs[i];
            }

            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // return the output neuron state 
            return layers[layers.Length - 1].neurons;
        }
    


        public void BackPropagate(float[] inputs, float[] expected)
        {
            // ensure proper neuron populations
            float[] output = FeedForward(inputs);

            cost = 0;
            float[] delta = new float[output.Length]; 
            for (int i = 0; i < delta.Length; i++)
            {
                // precalculate delta 
                delta[i] = output[i] - expected[i]; 

                // calculate cost of network 
                cost += (float)Math.Pow(delta[i], 2);
            }

            // setup 'gamma'
            float[][] gamma = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                gamma[i] = new float[layers[i].Size]; 
            }

            // calculate 'gamma' for the last layer
            layers[layers.Length - 1].CalculateGamma(delta, gamma[layers.Length - 1], output); 

            // calculate new weights and biases for the last layer in the network 
            for (int i = 0; i < layers[layers.Length - 1].Size; i++)
            {
                layers[layers.Length - 2].biases[i] -= gamma[layers.Length - 1][i] * learningRate;
                
                // apply some learning... move in direction of result using gamma 
                for (int j = 0; j < layers[layers.Length - 2].Size; j++)
                {
                    layers[layers.Length - 1].weights[i][j] 
                        -= 
                        gamma[layers.Length - 1][i]
                        *
                        layers[layers.Length - 2].neurons[j] 
                        * learningRate; 
                }
            }

            // now propagate backward 
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < layers[i].Size; j++)
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * layers[i + 1].weights[k][j];
                    }
                    
                    layers[i].CalculateGamma(gamma[i]); 
                }

                // modify bias to outputlayer 
                for (int j = 0; j < layers[i].Size; j++)
                {
                    // update bias 
                    layers[i].biases[j] -= gamma[i][j] * learningRate;

                    // modify weight to this layers inputs
                    for (int k = 0; k < layers[i - 1].Size; k++)
                    {
                        layers[i].weights[j][k] -= gamma[i][j] * layers[i - 1].neurons[k] * learningRate;
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
                for (int j = 0; j < layers[i].biases.Length; j++)
                {                           
                    layers[i].biases[j] = ((Random.Shared.NextSingle() * high) <= 2) 
                        ?
                        layers[i].biases[j] += RandomRange(-val, val)
                        :
                        layers[i].biases[j];
                }
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].weights.Length; j++)
                {
                    for (int k = 0; k < layers[i].weights[j].Length; k++)
                    {
                        layers[i].weights[j][k] = (RandomRange(0f, high) <= 2) ? layers[i].weights[j][k] += RandomRange(-val, val) : layers[i].weights[j][k];
                    }
                }
            }
        }

        public void DeepClone(NeuralNetwork into)
        {
            into.cost = cost;
            into.learningRate = learningRate;
            into.fitness = fitness;

            for (int i = 0; i < layers.Length; i++)
            {
                for(int j = 0; j < layers[i].neurons.Length; j++)
                {
                    into.layers[i].neurons[j] = layers[i].neurons[j]; 
                }

                if (!layers[i].IsInput)
                {
                    for (int j = 0; j < layers[i].biases.Length; j++)
                    {
                        into.layers[i].biases[j] = layers[i].biases[j];
                    }
                    for (int j = 0; j < layers[i].weights.Length; j++)
                    {
                        for (int k = 0; k < layers[i].weights[j].Length; k++)
                        {
                            into.layers[i].weights[j][k] = layers[i].weights[j][k]; 
                        }
                    }
                }
            }
        }

        public NeuralNetwork DeepCopy()
        {
            return new NeuralNetwork(this);            
        }

        public void Load(string path)
        {
            TextReader tr = new StreamReader(path);
            int NumberOfLines = (int)new FileInfo(path).Length;
            string[] ListLines = new string[NumberOfLines];
            int index = 1;
            for (int i = 1; i < NumberOfLines; i++)
            {
                ListLines[i] = tr.ReadLine();
            }
            tr.Close();
            if (new FileInfo(path).Length > 0)
            {
                
                for (int i = 1; i < LayerCount; i++)
                {
                    for (int j = 0; j < layers[i].biases.Length; j++)
                    {
                        layers[i].biases[j] = float.Parse(ListLines[index]);
                        index++;
                    }
                }

                for (int i = 1; i < LayerCount; i++)
                {
                    for (int j = 0; j < layers[i].weights.Length; j++)
                    {
                        for (int k = 0; k < layers[i].weights[j].Length; k++)
                        {
                            layers[i].weights[j][k] = float.Parse(ListLines[index]);
                            index++;
                        }
                    }
                }
            }
        }

        public void Save(string path)
        {
            File.Create(path).Close();
            StreamWriter writer = new StreamWriter(path, true);

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].biases.Length; j++)
                {
                    writer.WriteLine(layers[i].biases[j]);
                }
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].weights.Length; j++)
                {
                    for (int k = 0; k < layers[i].weights[j].Length; k++)
                    {
                        writer.WriteLine(layers[i].weights[j][k]);
                    }
                }
            }
            writer.Close();
        }

        public int CompareTo(object? obj)
        {
            return CompareTo((obj as NeuralNetwork)!);
        }
        public int CompareTo(NeuralNetwork other)
        {
            if (other == null) return 1;

            if (fitness > other.fitness)
                return 1;
            else if (fitness < other.fitness)
                return -1;
            else
                return 0;
        }

    }


}

