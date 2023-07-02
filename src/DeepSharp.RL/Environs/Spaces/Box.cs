using FluentAssertions;

namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     A T-dimensional box that contains every point in the action space.
    /// </summary>
    public class Box : DigitalSpace
    {
        public Box(float low, float high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low), torch.full(shape, high), shape, torch.ScalarType.Float32, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(double low, double high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low), torch.full(shape, high), shape, torch.ScalarType.Float64, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(long low, long high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low), torch.full(shape, high), shape, torch.ScalarType.Int64, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(int low, int high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low, torch.ScalarType.Int32), torch.full(shape, high, torch.ScalarType.Int32), shape,
                torch.ScalarType.Int32, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(short low, short high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low), torch.full(shape, high), shape, torch.ScalarType.Int16, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(byte low, byte high, long[] shape, DeviceType deviceType = DeviceType.CUDA, long seed = 1) :
            base(torch.full(shape, low), torch.full(shape, high), shape, torch.ScalarType.Byte, deviceType, seed)
        {
            CoculateBounded();
        }

        public Box(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1) : base(low, high, shape, type, deviceType, seed)
        {
            CoculateBounded();
        }

        protected torch.Tensor BoundedBelow { get; private set; } = null!;
        protected torch.Tensor BoundedAbove { get; private set; } = null!;

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
                .sample(1).reshape(Shape)[unbounded].to_type(Type);

            sample[lowBounded] = (Low + torch.distributions.Exponential(torch.ones(Shape, torch.ScalarType.Float32))
                .sample(1)
                .reshape(Shape))[lowBounded].to_type(Type);

            sample[uppBounded] =
                (high - torch.distributions.Exponential(torch.ones(Shape, torch.ScalarType.Float32)).sample(1)
                    .reshape(Shape))[uppBounded].to_type(Type);

            sample[bounded] =
                torch.distributions
                    .Uniform(Low.to_type(torch.ScalarType.Float32), high.to_type(torch.ScalarType.Float32)).sample(1)
                    .reshape(Shape)[bounded]
                    .to_type(Type);

            return sample.to(Device);
        }

        public override void CheckType()
        {
            var acceptType = new[]
            {
                torch.ScalarType.Byte,
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