using System.Text;

namespace DeepSharp.Utility.Operations
{
    public class OpTensor
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

        public static string ToArrString(torch.Tensor tensor)
        {
            var array = tensor.data<float>().ToArray();
            return string.Join(",", array);
        }

        public static string ToLongArrString(torch.Tensor tensor)
        {
            var array = tensor.data<long>().ToArray();
            return string.Join(",", array);
        }
    }
}