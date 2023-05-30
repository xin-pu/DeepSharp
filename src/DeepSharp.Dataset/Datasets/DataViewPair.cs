namespace DeepSharp.Dataset
{
    public class DataViewPair
    {
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


        internal DataViewPair To(torch.Device device)
        {
            var res = new DataViewPair(Labels.to(device), Features.to(device));
            return res;
        }

        /// <summary>
        ///     Send DataViewPair to CPU device
        /// </summary>
        /// <returns></returns>
        public DataViewPair cpu()
        {
            return To(new torch.Device(DeviceType.CPU));
        }

        /// <summary>
        ///     Send DataViewPair to CPU device
        /// </summary>
        /// <returns></returns>
        public DataViewPair cuda()
        {
            return To(new torch.Device(DeviceType.CUDA));
        }

        public override string ToString()
        {
            var strbuild = new StringBuilder();
            strbuild.AppendLine($"{Labels}");
            strbuild.AppendLine($"{Features}");
            return strbuild.ToString();
        }
    }
}