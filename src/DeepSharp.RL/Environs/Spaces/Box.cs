using FluentAssertions;

namespace DeepSharp.RL.Environs.Spaces
{
    public class Box : DigitalSpace
    {
        public Box(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1) : base(low, high, shape, type, deviceType, seed)
        {
            CoculateBounded();
        }

        public torch.Tensor BoundedBelow { get; private set; } = null!;
        public torch.Tensor BoundedAbove { get; private set; } = null!;

        public override torch.Tensor Sample()
        {
            var unbounded = ~BoundedBelow & ~BoundedAbove;
            var uppBounded = ~BoundedBelow & BoundedAbove;
            var lowBounded = BoundedBelow & ~BoundedAbove;
            var bounded = BoundedBelow & BoundedAbove;


            var high = Type.ToString().StartsWith("F") ? High : High + 1;
            var sample = torch.empty(Shape, Type);

            sample[unbounded] = torch.distributions.Normal(torch.zeros(Shape, torch.ScalarType.Float32),
                    torch.ones(Shape, torch.ScalarType.Float32))
                .sample(1).reshape(Shape)[unbounded];

            sample[lowBounded] = (Low + torch.distributions.Exponential(torch.ones(Shape, torch.ScalarType.Float32))
                .sample(1)
                .reshape(Shape))[lowBounded];

            sample[uppBounded] =
                (high - torch.distributions.Exponential(torch.ones(Shape, torch.ScalarType.Float32)).sample(1)
                    .reshape(Shape))[uppBounded];

            sample[bounded] = torch.distributions.Uniform(Low, high).sample(1).reshape(Shape)[bounded]
                .to_type(Type);

            return sample.to(Device);
        }

        public override void CheckType()
        {
            var acceptType = new[]
            {
                torch.ScalarType.Int8,
                torch.ScalarType.Int16,
                torch.ScalarType.Int32,
                torch.ScalarType.Int64,
                torch.ScalarType.Float32,
                torch.ScalarType.Float64
            };
            Type.Should().BeOneOf(acceptType, $"Disperse accept Type in {string.Join(",", acceptType)}");
        }

        private void CoculateBounded()
        {
            Low.shape.Should().Equal(Shape);
            High.shape.Should().Equal(Shape);
            BoundedBelow = Low > torch.tensor(double.NegativeInfinity);
            BoundedAbove = High < torch.tensor(double.PositiveInfinity);
        }
    }
}