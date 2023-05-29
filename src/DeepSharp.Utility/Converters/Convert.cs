using DeepSharp.Utility.Operations;

namespace DeepSharp.Utility.Converters
{
    public class Convert
    {
        /// <summary>
        ///     Convert Mat to Tensor
        /// </summary>
        /// <param name="mat"></param>
        /// <returns></returns>
        public static torch.Tensor ToTensor(Mat mat)
        {
            var dims = OpMat.GetDims(mat);
            mat.GetArray(out float[] d);
            var original = torch.from_array(d);
            var final = original.reshape(dims);
            return final;
        }


        /// <summary>
        ///     转换为 Array
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static Array ToArray(torch.Tensor tensor)
        {
            switch (tensor.dtype)
            {
                case torch.ScalarType.Byte:
                    return tensor.data<sbyte>().ToNDArray();

                case torch.ScalarType.Int8:
                    return tensor.data<sbyte>().ToNDArray();

                case torch.ScalarType.Int16:
                    return tensor.data<short>().ToNDArray();

                case torch.ScalarType.Int32:
                    return tensor.data<int>().ToNDArray();

                case torch.ScalarType.Int64:
                    return tensor.data<long>().ToNDArray();

                case torch.ScalarType.Float16:
                case torch.ScalarType.BFloat16:
                case torch.ScalarType.Float32:
                    return tensor.data<float>().ToNDArray();
                case torch.ScalarType.Float64:
                    return tensor.data<double>().ToNDArray();

                case torch.ScalarType.Bool:
                    return tensor.data<bool>().ToNDArray();

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}