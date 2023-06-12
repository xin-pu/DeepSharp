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
            long seed = 471) : base(shape, type, deviceType, seed)
        {
            CheckParameters(low, high);
            Low = low;
            High = high;
        }

        public torch.Tensor Low { get; }
        public torch.Tensor High { get; }


        /// <summary>
        ///     Generates a tensor whose shape and type are consistent with the space definition.
        /// </summary>
        /// <returns></returns>
        public override torch.Tensor Generate()
        {
            return (torch.zeros(Shape, Type) + Low).to(Device);
        }

        private void CheckParameters(torch.Tensor low, torch.Tensor high)
        {
            low.Should().NotBeNull();
            high.Should().NotBeNull();
            torch.all(low < high).Equals(torch.tensor(true)).Should().Be(true);
        }
    }
}