using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NSS.Imaging
{

    public class MapObject
    {
        public uint Id;
        public uint Density;

        public RECT Region;
        public QUAD Quad;
        public RECT InnerRegion;

        public MapObjectType Type;

        /// <summary>
        /// not subtracting the inner rect from the surface... 
        /// </summary>
        public double WeightedDensity
        {
            get
            {
                return (double)Density / (double)Region.Surface;
            }
        }

        public MapObject Parent = null;
        public List<MapObject> Children = null;
        public int ChildCount
        {
            get
            {
                if (Children == null) return 0;
                return Children.Count;
            }
        }

        public object Tag = null;

        public static bool IsNull(MapObject obj)
        {
            return obj == null;
        }

        #region Sort Comparers 

        public static int CompareOnWeightedDensity(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    double d1 = (double)obj1.Density / obj1.Region.Surface;
                    double d2 = (double)obj2.Density / obj2.Region.Surface;

                    if (d1 > d2)
                        return 1;
                    else
                     if (d1 < d2)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnDensity(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Density > obj2.Density)
                        return 1;
                    else
                     if (obj1.Density < obj2.Density)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnSurface(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Region.Surface > obj2.Region.Surface)
                        return 1;
                    else
                     if (obj1.Region.Surface < obj2.Region.Surface)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnId(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Id > obj2.Id)
                        return 1;
                    else
                     if (obj1.Id < obj2.Id)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnWidth(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Region.Width > obj2.Region.Width)
                        return 1;
                    else
                     if (obj1.Region.Width < obj2.Region.Width)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnHeight(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Region.Height > obj2.Region.Height)
                        return 1;
                    else
                     if (obj1.Region.Height < obj2.Region.Height)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        public static int CompareOnTop(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    if (obj1.Region.Top > obj2.Region.Top)
                        return 1;
                    else
                     if (obj1.Region.Top < obj2.Region.Top)
                        return -1;
                    else
                        return 0;
                }
            }
        }

        /// <summary>
        /// sort on position from origen 
        /// </summary>
        /// <param name="obj1"></param>
        /// <param name="obj2"></param>
        /// <returns></returns>
        public static int CompareOnPosition(MapObject obj1, MapObject obj2)
        {
            if (obj1 == null)
            {
                if (obj2 == null)
                {
                    return 0;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (obj2 == null)
                {
                    return 1;
                }
                else
                {
                    double d1 = obj1.Region.CenterX * obj1.Region.CenterX + obj1.Region.CenterY * obj1.Region.CenterY;
                    double d2 = obj2.Region.CenterX * obj2.Region.CenterX + obj2.Region.CenterY * obj2.Region.CenterY;

                    if (d1 > d2)
                        return 1;
                    else
                     if (d1 < d2)
                        return -1;
                    else
                        return 0;
                }
            }
        }


        #endregion
    }

    public class MapFrameObject : MapObject
    {
        public int GM_Left = -1;
        public int GM_Right = -1;
        public int GM_Top = -1;
        public int GM_Bottom = -1;
        public bool GM_Calculated = false;

        public float TopSlope = 0f;
        public float LeftSlope = 0f;
        public float BottomSlope = 0f;
        public float RightSlope = 0f;

        public static MapFrameObject FromObject(MapObject obj)
        {
            if (obj == null) return null;

            MapFrameObject o = new MapFrameObject();
            o.Id = obj.Id;
            o.Region = obj.Region;
            o.InnerRegion = obj.InnerRegion;
            o.Density = obj.Density;
            o.Children = obj.Children;
            o.Parent = obj.Parent;
            o.Tag = obj.Tag;
            o.Type = MapObjectType.Frame;

            return o;
        }
    }


    public class MapSolidObject : MapObject
    {
        public static MapSolidObject FromObject(MapObject obj)
        {
            if (obj == null) return null;

            MapSolidObject o = new MapSolidObject();
            o.Id = obj.Id;
            o.Region = obj.Region;
            o.InnerRegion = obj.Region;
            o.Density = obj.Density;
            o.Children = obj.Children;
            o.Parent = obj.Parent;
            o.Tag = obj.Tag;
            o.Type = MapObjectType.Solid;

            return o;
        }
    }
}
