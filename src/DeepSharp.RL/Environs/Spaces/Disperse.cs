namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     Discrete(2)            # {0, 1}
    ///     Discrete(3, start=-1)  # {-1, 0, 1}
    /// </summary>
    public class Disperse : DigitalSpace
    {
        public Disperse(long length, long start, torch.ScalarType dtype = torch.ScalarType.Int64,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(torch.tensor(new[] {start}, dtype),
                torch.tensor(new[] {start + length - 1}, dtype),
                new long[] {1},
                dtype, deviceType, seed)
        {
        }

        public Disperse(long length, torch.ScalarType dtype = torch.ScalarType.Int64,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : this(length, 0, dtype, deviceType, seed)
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