using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    static public class BinaryReaderExtensions
    {
        static public LayerActivationFunction ReadLayerActivationFunction(this BinaryReader r)
        {
            return (LayerActivationFunction)r.ReadByte();
        }
        static public LayerInitializer ReadLayerInitializer(this BinaryReader r)
        {
            return (LayerInitializer)r.ReadByte();
        }
        static public Layer ReadLayer(this BinaryReader r)
        {
            int size = r.ReadInt32();
            int previousSize = r.ReadInt32();

            LayerActivationFunction act = r.ReadLayerActivationFunction();
            LayerInitializer biasInit = r.ReadLayerInitializer();
            LayerInitializer weightInit = r.ReadLayerInitializer();

            Layer layer = NeuralNetwork.CreateLayer(size, previousSize, act, weightInit, biasInit, true);

            if (previousSize > 0) // == IsInput 
            {
                for (int i = 0; i < size; i++)
                {
                    layer.Biases[i] = r.ReadSingle();
                }
                for (int i = 0; i < layer.Size; i++)
                {
                    for (int j = 0; j < layer.PreviousSize; j++)
                    {
                        layer.Weights[i][j] = r.ReadSingle();
                    }
                }
            }

            return layer;
        }

        static public NeuralNetwork ReadNeuralNetwork(this BinaryReader r)
        {
            return new NeuralNetwork(r);
        }
    }
}
