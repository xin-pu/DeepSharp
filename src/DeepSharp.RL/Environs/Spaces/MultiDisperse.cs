namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     A list of possible actions, where each timestep only one action of each discrete set can be used.
    /// </summary>
    public class MultiDisperse : DigitalSpace
    {
        public MultiDisperse(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(low, high, shape, type, deviceType, seed)
        {
        }

        public MultiDisperse(long low, long high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(low, high, shape, type, deviceType, seed)
        {
        }

        public override torch.Tensor Sample()
        {
            var high = High + 1;

            var sample = torch.distributions.Uniform(Low.to_type(torch.ScalarType.Float32),
                    high.to_type(torch.ScalarType.Float32), Generator)
                .sample(1)
                .reshape(Shape)
                .to_type(Type);

            return sample.to(Device);
        }
    }
}