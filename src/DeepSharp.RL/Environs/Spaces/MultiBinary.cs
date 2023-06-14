namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     A list of possible actions, where each timestep any of the actions can be used in any combination.
    /// </summary>
    public class MultiBinary : DigitalSpace
    {
        public MultiBinary(long[] shape, torch.ScalarType type = torch.ScalarType.Int32,
            DeviceType deviceType = DeviceType.CUDA, long seed = 471) : base(0, 1, shape, type, deviceType, seed)
        {
        }

        public MultiBinary(long shape, torch.ScalarType type = torch.ScalarType.Int32,
            DeviceType deviceType = DeviceType.CUDA, long seed = 471) : base(0, 1, new[] {shape}, type, deviceType,
            seed)
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