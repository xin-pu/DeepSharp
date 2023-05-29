using System.Text;

namespace DeepSharp.Utility
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
    }
}