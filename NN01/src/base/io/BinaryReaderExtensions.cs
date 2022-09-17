using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{
    static public class BinaryReaderExtensions
    {
        static public LayerType ReadLayerActivationFunction(this BinaryReader r)
        {
            return (LayerType)r.ReadByte();
        }
        static public LayerInitializationType ReadLayerInitializer(this BinaryReader r)
        {
            return (LayerInitializationType)r.ReadByte();
        }
        static public Layer ReadLayer(this BinaryReader r)
        {
            int size = r.ReadInt32();
            int previousSize = r.ReadInt32();

            LayerType act = r.ReadLayerActivationFunction();
            LayerInitializationType biasInit = r.ReadLayerInitializer();
            LayerInitializationType weightInit = r.ReadLayerInitializer();

            Layer layer = NeuralNetwork.CreateLayer(size, previousSize, act, true);
         /*
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
                        layer.Weights[i,j] = r.ReadSingle();
                    }
                }
            }
           */
            return layer;
        }

        static public NeuralNetwork ReadNeuralNetwork(this BinaryReader r)
        {
            return new NeuralNetwork(r);
        }
    }
}
