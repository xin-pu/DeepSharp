using FluentAssertions;

namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     Discrete(2)            # {0, 1}
    ///     Discrete(3, start=-1)  # {-1, 0, 1}
    /// </summary>
    public class Disperse : DigitalSpace
    {
        public long N { get; init; }

        public Disperse(long length, long start, torch.ScalarType dtype = torch.ScalarType.Int64,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(torch.tensor(new[] {start}, dtype).to(new torch.Device(deviceType)),
                torch.tensor(new[] {start + length - 1}, dtype).to(new torch.Device(deviceType)),
                new long[] {1},
                dtype, deviceType, seed)
        {
            N = length;
        }

        public Disperse(long length, torch.ScalarType dtype = torch.ScalarType.Int64,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : this(length, 0, dtype, deviceType, seed)
        {
        }

        public override torch.Tensor Sample()
        {
            var device = new torch.Device(DeviceType);
            var high = High + 1;
            var sample = torch.randint(Low.item<long>(), high.item<long>(), Shape, Type, device);
            return sample;
        }

        public override void CheckType()
        {
            var acceptType = new[]
            {
                torch.ScalarType.Int8,
                torch.ScalarType.Int16,
                torch.ScalarType.Int32,
                torch.ScalarType.Int64
            };
            Type.Should().BeOneOf(acceptType, $"Disperse accept Type in {string.Join(",", acceptType)}");
        }
    }
}