using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{                 
    public enum LayerInitializationType : byte
    {
        /// <summary>
        /// default to activation default
        /// </summary>
        Default = 0,

        /// <summary>
        /// used in linear modes
        /// </summary>
        dot01,
        Zeros,
        Ones,

        /// <summary>
        /// distribute 1 evenly over all elements
        /// </summary>
        Uniform,

        /// <summary>
        /// Rectified Linear modes 
        /// </summary>
        Random,

        Normal,

        /// <summary>
        /// Genarally used for for tanh activation 
        /// </summary>
        HeNormal,

        Glorot,

        Xavier, 

        XavierNormalized



        /// <summary>
        /// == Xavier 
        /// </summary>
        // Glorot 
    
    }
}
