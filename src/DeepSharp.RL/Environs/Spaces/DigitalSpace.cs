using FluentAssertions;

namespace DeepSharp.RL.Environs.Spaces
{
    public abstract class DigitalSpace : Space
    {
        protected DigitalSpace(
            torch.Tensor low,
            torch.Tensor high,
            long[] shape,
            torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA,
            long seed = 1) : base(shape, type, deviceType, seed)
        {
            CheckParameters(low, high);
            Low = low;
            High = high;
        }

        public torch.Tensor Low { get; private init; }
        public torch.Tensor High { get; private init; }


        private void CheckParameters(torch.Tensor low, torch.Tensor high)
        {
            low.Should().NotBeNull();
            high.Should().NotBeNull();
            torch.all(low < high).Equals(torch.tensor(true)).Should().Be(true);
        }

        public override torch.Tensor Sample()
        {
            throw new NotImplementedException();
        }
    }
}