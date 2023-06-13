namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     一维 离散的动作空间， 采样为编码的动作序号
    ///     Discrete(2)             # {0, 1}
    ///     Discrete(3)             # {0, 1, 2}
    /// </summary>
    public class Disperse : DigitalSpace
    {
        public Disperse(long length, torch.ScalarType dtype = torch.ScalarType.Int64,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(0, 0 + length - 1, new long[] {1}, dtype, deviceType, seed)
        {
            N = length;
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