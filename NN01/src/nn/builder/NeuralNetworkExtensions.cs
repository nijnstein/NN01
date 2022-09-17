using NSS;
using System.Diagnostics;
using System.Drawing.Printing;

namespace NN01
{
    public static class NeuralNetworkExtensions
    {
        static public NeuralNetworkBuilder WithLayer(this NeuralNetworkBuilder builder, int layerSize, LayerType activationFunction)
        {
            Debug.Assert(layerSize > 0); 
            Debug.Assert(builder.stack.Any(), "at least an input must be defined");
            // Debug.Assert(builder.stack.Last().ActivationType != LayerActivationFunction.Softmax);  

            Layer layer = NeuralNetwork.CreateLayer(layerSize, builder.stack.Last().Size, activationFunction, true, null);
            builder.stack.Add(layer);
            return builder;
        }

        static public NeuralNetworkBuilder Dropout(this NeuralNetworkBuilder builder, float dropoutFactor)
        {
            Debug.Assert(builder.stack.Any(), "at least an input must be defined");
            Layer layer = new Dropout(builder.stack.Last().Size, dropoutFactor);
            builder.stack.Add(layer);
            return builder;
        }

        static public NeuralNetworkBuilder ReLU(this NeuralNetworkBuilder builder, int layerSize)
        {
            return WithLayer(builder, layerSize, LayerType.ReLU);
        }

        static public NeuralNetworkBuilder LeakyReLU(this NeuralNetworkBuilder builder, int layerSize)
        {
            return WithLayer(builder, layerSize, LayerType.LeakyReLU);
        }

        static public NeuralNetworkBuilder TanH(this NeuralNetworkBuilder builder, int layerSize)
        {
            return WithLayer(builder, layerSize, LayerType.Tanh);
        }

        static public NeuralNetworkBuilder Swish(this NeuralNetworkBuilder builder, int layerSize)
        {
            return WithLayer(builder, layerSize, LayerType.Swish);
        }

        static public NeuralNetworkBuilder Sigmoid(this NeuralNetworkBuilder builder, int layerSize)
        {
            return WithLayer(builder, layerSize, LayerType.Sigmoid);
        }

        static public NeuralNetworkBuilder Softmax(this NeuralNetworkBuilder builder, bool enabled = true)
        {
            if (enabled)
            {
                Debug.Assert(builder != null && builder.stack != null);
                Layer prev = builder!.stack!.LastOrDefault()!;

                Debug.Assert(prev != null);
                Debug.Assert(!prev.IsInput, "cannot have an output directly connected on the input layer");

                int classCount = prev.Size;
                Debug.Assert(classCount > 0);

                builder.stack.Add(new SoftmaxLayer(classCount));
            }
            return builder;
        }

        static public NeuralNetworkBuilder Convolve2D(this NeuralNetworkBuilder builder, Size2D kernelSize)
        {
            Debug.Assert(builder != null && builder.stack != null);
            Layer prev = builder!.stack!.LastOrDefault()!;

            Debug.Assert(prev is Layer2D, "previous layer must be 2 dimensional");
            Layer2D prev2D = prev as Layer2D; 

            Convolve2D layer = new Convolve2D(prev2D.Size2D, prev2D.Size2D, kernelSize);
            builder.stack.Add(layer);
            return builder; 
        }

        static public NeuralNetwork Build(this NeuralNetworkBuilder builder, IRandom? random = null)
        {
            Debug.Assert(builder.stack.Count > 2, "the network must define at least 3 layers: (input, hidden and output)"); 
            NeuralNetwork network = new NeuralNetwork(builder.stack.ToArray());
            network.InitializeLayers(random); 
            return network;
        }
    }
}

