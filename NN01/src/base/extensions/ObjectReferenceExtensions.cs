using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NN01
{

    public static class ObjectReferenceExtensions
    {
        private static object mutualObject;
        private static ObjectReinterpreter reinterpreter;

        static ObjectReferenceExtensions()
        {
            mutualObject = new object();
            reinterpreter = new ObjectReinterpreter();
            reinterpreter.AsObject = new ObjectWrapper();
        }

        public static IntPtr GetAddress(this object obj)
        {
            lock (ObjectReferenceExtensions.mutualObject)
            {
                reinterpreter.AsObject.Object = obj;
                IntPtr address = reinterpreter.AsIntPtr.Value;
                reinterpreter.AsObject.Object = null;
                return address;
            }
        }

        public static T GetInstance<T>(this IntPtr address)
        {
            lock (mutualObject)
            {
                reinterpreter.AsIntPtr.Value = address;
                T obj = (T)reinterpreter.AsObject.Object;
                reinterpreter.AsObject.Object = null;
                return obj;
            }
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct ObjectReinterpreter
        {
            [FieldOffset(0)] public ObjectWrapper AsObject;
            [FieldOffset(0)] public IntPtrWrapper AsIntPtr;
        }

        private class ObjectWrapper
        {
            public object Object;
        }

        private class IntPtrWrapper
        {
            public IntPtr Value;
        }
    }
}
