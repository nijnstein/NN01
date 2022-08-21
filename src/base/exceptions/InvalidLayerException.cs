using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NN01 
{
    public class InvalidLayerException : Exception
    {
        public InvalidLayerException()
        {
        }

        public InvalidLayerException(string? message) : base(message)
        {
        }

        public InvalidLayerException(string? message, Exception? innerException) : base(message, innerException)
        {
        }

        protected InvalidLayerException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}
