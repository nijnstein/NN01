﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{                 
    public enum LayerInitializer
    {
        /// <summary>
        /// used in linear modes
        /// </summary>
        Zeros,
        Ones,

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
