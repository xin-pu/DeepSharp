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


        #region Box

        [Fact]
        public void CreateFloatBox()
        {
            var box = new Box(0f, 1f, new long[] {2, 2});
            var r = box.Sample();
            Print(r);
        }

        [Fact]
        public void CreateDoubleBox()
        {
            var box = new Box(0d, 1d, new long[] {2, 2});
            var r = box.Sample();
            Print(r);
        }


        [Fact]
        public void CreateInt32Box()
        {
            var box = new Box(1, 5, new long[] {10});
            var r = box.Sample();
            Print(r);
        }

        [Fact]
        public void CreateInt64Box()
        {
            var box = new Box(1L, 5L, new long[] {10});
            var r = box.Sample();
            Print(r);
        }

        [Fact]
        public void CreateByteBox()
        {
            var box = new Box((byte) 0, (byte) 1, new long[] {10});
            var r = box.Sample();
            Print(r);
        }

        [Fact]
        public void CreateInt16Box()
        {
            var box = new Box((short) 1, (short) 5, new long[] {10});
            var r = box.Sample();
            Print(r);
        }

        #endregion


        #region MyRegion

        #region Other Space

        [Fact]
        public void CreateMultiDisperse()
        {
            var multiDisperse = new MultiDisperse(
                torch.tensor(new long[] {0, 0}),
                torch.tensor(new long[] {3, 4}),
                new long[2], torch.ScalarType.Int32
            );
            var r = multiDisperse.Sample();
            Print(r);
            Print(multiDisperse);
        }


        [Fact]
        public void CreateBinary()
        {
            var binary = new Binary(torch.ScalarType.Int64);
            var r = binary.Sample();
            Print(r);
            Print(binary);
        }


        [Fact]
        public void CreateMultiBinary()
        {
            var binary = new MultiBinary(2);
            var r = binary.Sample();
            Print(r);
            Print(binary);
        }

        #endregion

        #endregion
    }
}