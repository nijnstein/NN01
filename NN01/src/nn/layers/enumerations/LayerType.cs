using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public enum LayerType : byte
    {
        Sigmoid,

        Tanh,

        /// <summary>
        /// Rectified Linear Unit 
        /// </summary>
        ReLU,

        /// <summary>
        /// Leaky Rectified Linear Unit
        /// </summary>
        LeakyReLU,
 
        /// <summary>
        /// swish by google 
        /// </summary>
        Swish,

        /// <summary>
        /// Binary Step
        /// </summary>
        Binary,
             
        None, 

        Softmax,
        
        Dropout
    }
}
