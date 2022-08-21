﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public class InvalidFormatException : Exception
    {
        public InvalidFormatException()
        {
        }

        public InvalidFormatException(string? message) : base(message)
        {
        }

        public InvalidFormatException(string? message, Exception? innerException) : base(message, innerException)
        {
        }

        protected InvalidFormatException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}
