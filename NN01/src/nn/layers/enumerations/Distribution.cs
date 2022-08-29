using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{                 
    public enum Distribution : byte
    {
        /// <summary>
        /// default to activation default
        /// </summary>
        Default = 0,

        /// <summary>
        /// used in linear modes
        /// </summary>
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

        Gaussian,

        Normal,

        /// <summary>
        /// Genarally used for for tanh activation 
        /// </summary>
        HeNormal,

        /// <summary>
        /// == Xavier 
        /// </summary>
        // Glorot 
    
    }
}
