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
            CoculateBounded();
        }

        public torch.Tensor Low { get; }
        public torch.Tensor High { get; }
        public torch.Tensor BoundedBelow { get; private set; } = null!;
        public torch.Tensor BoundedAbove { get; private set; } = null!;

        /// <summary>
        ///     Generates a tensor whose shape and type are consistent with the space definition.
        /// </summary>
        /// <returns></returns>
        public override torch.Tensor Generate()
        {
            return torch.zeros(Shape, Type, Device) + Low;
        }

        private void CheckParameters(torch.Tensor low, torch.Tensor high)
        {
            low.Should().NotBeNull();
            high.Should().NotBeNull();
            torch.all(low < high).Equals(torch.tensor(true).to(Device)).Should().Be(true);
        }


        private void CoculateBounded()
        {
            BoundedBelow = Low > torch.tensor(double.NegativeInfinity, device: Device);
            BoundedAbove = High < torch.tensor(double.PositiveInfinity, device: Device);
        }
    }
}