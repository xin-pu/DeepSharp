using DeepSharp.RL.Environs.Spaces;
using FluentAssertions;

namespace TorchSharpTest.RLTest
{
    public class SpaceTest : AbstractTest
    {
        public SpaceTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        #region Disperse Test

        [Fact]
        public void DisperseCons()
        {
            var disperse = new Disperse(5, 1);
            var s = disperse.Generate();
            Print(s);

            var disperse2 = new Disperse(2);
            s = disperse2.Generate();
            Print(s);
        }

        [Fact]
        public void DisperseGenerate()
        {
            var disperse = new Disperse(5, 1, torch.ScalarType.Int8);
            var one = disperse.Generate();
            Print(one);
            one.dtype.Should().Be(torch.ScalarType.Int8);

            disperse = new Disperse(5, 1, torch.ScalarType.Int16);
            one = disperse.Generate();
            Print(one);
            one.dtype.Should().Be(torch.ScalarType.Int16);

            disperse = new Disperse(5, 1, torch.ScalarType.Int32);
            one = disperse.Generate();
            Print(one);
            one.dtype.Should().Be(torch.ScalarType.Int32);

            disperse = new Disperse(5, 1);
            one = disperse.Generate();
            Print(one);
            one.dtype.Should().Be(torch.ScalarType.Int64);
        }

        [Fact]
        public void DisperseDevice()
        {
            var disperse = new Disperse(5, 1, deviceType: DeviceType.CUDA);
            var one = disperse.Generate();
            Print(one);
            one.device_type.Should().Be(DeviceType.CUDA);

            disperse = new Disperse(5, 1, deviceType: DeviceType.CPU);
            one = disperse.Generate();
            Print(one);
            one.device_type.Should().Be(DeviceType.CPU);
        }


        [Fact]
        public void DisperseSample()
        {
            var a = new Disperse(5, 1);
            foreach (var _ in Enumerable.Repeat(0, 10))
            {
                var data = a.Sample();
                Print(data);
                data.ToInt64().Should().BeInRange(1, 5);
            }
        }

        #endregion


        #region Binary

        [Fact]
        public void CreateBinary()
        {
            var binary = new Binary();
            var r = binary.Sample();
            Print(r);
        }

        #endregion


        #region Box

        [Fact]
        public void CreateInt32Box()
        {
            var binary = new Box(torch.tensor(new[,] {{0, float.NegativeInfinity}, {100, float.NegativeInfinity}}),
                torch.tensor(new[,] {{5, 5}, {float.PositiveInfinity, float.PositiveInfinity}}),
                new long[] {2, 2},
                torch.ScalarType.Int32);
            var r = binary.Sample();
            Print(r);
        }


        [Fact]
        public void CreateFloatBox()
        {
            var binary = new Box(torch.tensor(new[,] {{0, float.NegativeInfinity}, {100, float.NegativeInfinity}}),
                torch.tensor(new[,] {{1, 5}, {float.PositiveInfinity, float.PositiveInfinity}}),
                new long[] {2, 2},
                torch.ScalarType.Float32);
            var r = binary.Sample();
            Print(r);
        }


        [Fact]
        public void CreateDoubleBox()
        {
            var binary = new Box(torch.tensor(new[,] {{0, float.NegativeInfinity}, {100, float.NegativeInfinity}}),
                torch.tensor(new[,] {{1, 5}, {float.PositiveInfinity, float.PositiveInfinity}}),
                new long[] {2, 2},
                torch.ScalarType.Float64);
            var r = binary.Sample();
            Print(r);
        }

        #endregion
    }
}