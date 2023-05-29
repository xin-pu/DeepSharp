namespace DeepSharp.Dataset
{
    /// <summary>
    ///     包含批次信息的 数据集 张量对象
    /// </summary>
    public class DataViewPair
    {
        /// <summary>
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="features"></param>
        public DataViewPair(torch.Tensor labels, torch.Tensor features)
        {
            Labels = labels;
            Features = features;
        }

        public DataViewPair(IEnumerable<torch.Tensor> labels, IEnumerable<torch.Tensor> features)
        {
            var labelsArray = labels.ToArray();
            var featuresArray = features.ToArray();
            Labels = torch.vstack(labelsArray);
            Features = torch.vstack(featuresArray);
        }


        public torch.Tensor Labels { set; get; }
        public torch.Tensor Features { set; get; }

        public override string ToString()
        {
            var strbuild = new StringBuilder();
            strbuild.AppendLine($"{Labels}");
            strbuild.AppendLine($"{Features}");
            return strbuild.ToString();
        }

        public DataViewPair To(torch.Device device)
        {
            var res = new DataViewPair(Labels.to(device), Features.to(device));
            return res;
        }
    }
}