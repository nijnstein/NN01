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

        public float Fitness = 0;
        public float Cost = 0;

        public Layer this[int index]
        {
            get
            {
                if (layers == null) throw new NotInitializedException();
                if (index < 0 || index >= layers.Length) throw new IndexOutOfRangeException($"index {index} is not a valid layer index, layercount = {layers.Length}"); 
                return layers[index];
            }
        }

        public int LayerCount => layers == null ? 0 : layers.Length;

        public Layer Input => this[0];
        public Layer Output => this[layers.Length - 1];

        public NeuralNetwork(int[] layerSizes, LayerActivationFunction[] activations)
        {
            if (layerSizes == null) throw new ArgumentNullException("layerSizes");
            if (activations == null) throw new ArgumentNullException("activations");
            if (layerSizes.Length <= 2 || activations.Length <= 1 || layerSizes.Length != activations.Length + 1) throw new ArgumentOutOfRangeException("there must be at least 3 layers and activations must be 1 index shorter as they sit between layers");

            layers = new Layer[layerSizes.Length];

            layers[0] = CreateLayer(layerSizes[0]);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i] = CreateLayer(layerSizes[i], layerSizes[i - 1] , activations[i - 1]);
            }
        }
        public NeuralNetwork(Layer[] layers)
        {
            if (layers == null) throw new ArgumentNullException("layers"); 
            this.layers = layers; 
        }

        public NeuralNetwork(Stream stream) : this(new BinaryReader(stream))
        {
        }

        public NeuralNetwork(BinaryReader r)
        {
            if (r == null) throw new ArgumentNullException("binaryreader r");

            string magic = r.ReadString();
            if (string.Compare(magic, "NN01", false) != 0) throw new InvalidFormatException("stream does not contain data in correct format for reading in a neural network (magic marker not present)");

            int layerCount = r.ReadInt32();

            Fitness = r.ReadSingle();
            Cost = r.ReadSingle();

            layers = new Layer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layers[i] = r.ReadLayer();
            }
        }

        public NeuralNetwork(NeuralNetwork other)
        {
            layers = new Layer[other.layers.Length];

            layers[0] = CreateLayer(other.layers[0].Size);

            for (int i = 1; i < other.layers.Length; i++)
            {
                layers[i] = CreateLayer(
                    other.layers[i].Size, 
                    other.layers[i - 1].Size,
                    other.layers[i].ActivationType,
                    other.layers[i].WeightInitializer,
                    other.layers[i].BiasInitializer,
                    true
                );
            }

            other.DeepClone(this);
        }
        
        internal static Layer CreateLayer(int size, int previousSize = 0, LayerActivationFunction activationType = LayerActivationFunction.None, LayerInitializer weightInit = LayerInitializer.Default, LayerInitializer biasInit = LayerInitializer.Default, bool skipInitializers = false)
        {
            if(previousSize == 0)
            {
                return new InputLayer(size); 
            }

            switch (activationType)
            {
                case LayerActivationFunction.ReLU:
                    return new ReLuLayer(size, previousSize, weightInit, biasInit, skipInitializers);

                case LayerActivationFunction.LeakyReLU:
                    return new LeakyReLuLayer(size, previousSize, weightInit, biasInit, skipInitializers);

                case LayerActivationFunction.Tanh:
                    return new TanhLayer(size, previousSize, weightInit, biasInit, skipInitializers);

                case LayerActivationFunction.Sigmoid:
                    return new SigmoidLayer(size, previousSize, weightInit, biasInit, skipInitializers);
                    
                case LayerActivationFunction.Swish:
                    return new SwishLayer(size, previousSize, weightInit, biasInit, skipInitializers);

                case LayerActivationFunction.Binary:
                    return new BinaryLayer(size, previousSize, weightInit, biasInit, skipInitializers);
            }

            throw new InvalidLayerException($"cannot create layer, size: {size}, previous size: {previousSize}, activation: {activationType}, weight: {weightInit}, bias: {biasInit}, skip init: {skipInitializers}"); 
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
                Input.Neurons[i] = inputs[i];
            }

            // propagate state through layers 
            for (int i = 1; i < layers.Length; i++)
            {
                // activate neurons in current layer from state of previous layer 
                layers[i].Activate(layers[i - 1]);
            }

            // return the output neuron state 
            return Output.Neurons;
        }

        public float[] FeedBackward(float[] outputs)
        {
            // set output|last layer 
            for (int i = 0; i < outputs.Length; i++)
            {
                Output.Neurons[i] = outputs[i];
            }

            // now propagate backwards
            for (int i = LayerCount - 1; i > 0; i--)
            {
                Layer previous = layers[i - 1];
                Layer current = layers[i]; 

                // TODO              BackPropagate(current.)
            }

            return Input.Neurons;
        }

        /// <summary>
        /// backtrack state through derivates of the activation minimizing cost/error
        /// </summary>
        public void BackPropagate(
            float[] inputs, float[] expected, float[][] gamma, 
            float weightLearningRate = 0.01f, float biasLearningRate = 0.01f, float momentum = 1f, float weightCost = 0.00001f)
        {
            // initialize neurons from input patterns
            FeedForward(inputs);

            Cost = 0;
            float[] delta = new float[Output.Neurons.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                // precalculate delta 
                delta[i] = Output.Neurons[i] - expected[i];

                // calculate cost of network 
                Cost += (float)Math.Pow(delta[i], 2);
            }

            // calculate 'gamma' for the last layer
            Output.CalculateGamma(delta, gamma[layers.Length - 1], Output.Neurons);

            // calculate new weights and biases for the last layer in the network 
            Output.Update(layers[layers.Length - 2], gamma[layers.Length - 1], weightLearningRate, biasLearningRate, momentum, weightCost);

            // now propagate backward 
            for (int i = layers.Length - 2; i > 0; i--)
            {
                // update gamma from layer weights and current gamma on output
                for (int j = 0; j < layers[i].Size; j++)
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * layers[i + 1].Weights[k][j];
                    }

                    layers[i].CalculateGamma(gamma[i], gamma[i], layers[i].Neurons);
                }

                // update layer weights and biases
                layers[i].Update(layers[i - 1], gamma[i], weightLearningRate, biasLearningRate, momentum, weightCost); 
            }
        }

        /// <summary>
        /// randomly mutate some of the worst networks
        /// </summary>
        public void Mutate(int high, float val) 
        {
            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].Biases.Length; j++)
                {
                    layers[i].Biases[j] = Random.Shared.NextSingle() * high <= 2
                        ?
                        layers[i].Biases[j] += Random.Shared.Range(-val, val)
                        :
                        layers[i].Biases[j];
                }
            }

            for (int i = 1; i < LayerCount; i++)
            {
                for (int j = 0; j < layers[i].Weights.Length; j++)
                {
                    for (int k = 0; k < layers[i].Weights[j].Length; k++)
                    {
                        layers[i].Weights[j][k] = Random.Shared.Range(0f, high) <= 2 ? layers[i].Weights[j][k] += Random.Shared.Range(-val, val) : layers[i].Weights[j][k];
                    }
                }
            }
        }

        public void DeepClone(NeuralNetwork into)
        {
            into.Cost = Cost;
            into.Fitness = Fitness;

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    into.layers[i].Neurons[j] = layers[i].Neurons[j];
                }

                if (!layers[i].IsInput)
                {
                    for (int j = 0; j < layers[i].Biases.Length; j++)
                    {
                        into.layers[i].Biases[j] = layers[i].Biases[j];
                    }
                    for (int j = 0; j < layers[i].Weights.Length; j++)
                    {
                        for (int k = 0; k < layers[i].Weights[j].Length; k++)
                        {
                            into.layers[i].Weights[j][k] = layers[i].Weights[j][k];
                        }
                    }
                }
            }
        }

        public void WriteTo(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException("stream");
            if (!stream.CanWrite) throw new IOException("cannot write to stream, it is non writable"); 

            using(BinaryWriter w = new BinaryWriter(stream))
            {
                w.Write(this); 
            }
        }

        public static NeuralNetwork ReadFrom(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException("stream");
            if (!stream.CanRead) throw new IOException("cannot read from stream, it is not readable");

            using(BinaryReader r = new BinaryReader(stream))
            {
                return r.ReadNeuralNetwork(); 
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

