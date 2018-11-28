//#include "float3_lib.h"
//
//
//__device__
//Float3 operator*(Float3 &a, float b)
//{
//    // TODO: use make_float as in (https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4)
//    Float3 res;
//    res.arr[0] = a.arr[0] * b;
//    res.arr[1] = a.arr[1] * b;
//    res.arr[2] = a.arr[2] * b;
//    return res;
//}
//
//
//__device__
//Float3 operator-(Float3 &a, Float3 &b)
//{
//    Float3 res;
//    res.arr[0] = a.arr[0] - b.arr[0];
//    res.arr[1] = a.arr[1] - b.arr[1];
//    res.arr[2] = a.arr[2] - b.arr[2];
//    return res;
//}
//
//
//__device__
//Float3 operator*(Float3 a, Float3 b)
//{
//    Float3 res;
//    res.arr[0] = a.arr[0] * b.arr[0];
//    res.arr[1] = a.arr[1] * b.arr[1];
//    res.arr[2] = a.arr[2] * b.arr[2];
//    return res;
//}
//
//__device__
//float reduceSum(Float3 &a) {
//    return a.arr[0] + a.arr[1] + a.arr[2];
//}
//
//__device__
//void resetValue(Float3 &a, float val) {
//    a.arr[0] = val;
//    a.arr[1] = val;
//    a.arr[2] = val;
//}
//
//__device__
//Float3& operator+=(Float3 &first, const Float3& sec) {
//    first.arr[0] += sec.arr[0];
//    first.arr[1] += sec.arr[1];
//    first.arr[2] += sec.arr[2];
//    return first;
//}