using System.Text;

namespace RL.Core.Utility
{
    public class TensorOp
    {
        public static string ToString(torch.Tensor tensor, int maxLengthShow = 4)
        {
            var dims = tensor.shape;
            var dtype = tensor.dtype;
            var array = tensor.data<float>().ToNDArray();

            var strBuild = new StringBuilder();


            strBuild.AppendLine($"[{string.Join(",", dims)}]\t{dtype}\t{tensor.device}");
            return strBuild.ToString();
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