namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     Todo
    /// </summary>
    public class MultiDisperse : DigitalSpace
    {
        public MultiDisperse(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(low, high, shape, type, deviceType, seed)
        {
        }

        public override torch.Tensor Sample()
        {
            var device = new torch.Device(DeviceType);
            var low = Low.to_type(torch.ScalarType.Int64).item<long>();
            var high = (High + 1).to_type(torch.ScalarType.Int64).item<long>();

            var sample = torch.randint(low, high, Shape, device: device).to_type(Type);

            return sample;
        }
    }
}