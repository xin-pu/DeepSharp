using DeepSharp.RL.Environs.Spaces;

namespace TorchSharpTest.RLTest.ModelTest
{
	public class SpaceTest : AbstractTest
	{
		public SpaceTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}


		#region Discrete Test

		[Fact]
		public void DiscreteCons()
		{
			var disperse = new Discrete(5);
			var s        = disperse.Generate();
			Print(s);

			var disperse2 = new Discrete(2);
			s = disperse2.Generate();
			Print(s);
		}

		[Fact]
		public void DiscreteGenerate()
		{
			var disperse = new Discrete(5, ScalarType.Int8);
			var one      = disperse.Generate();
			Print(one);
			one.dtype.Should().Be(ScalarType.Int8);

			disperse = new Discrete(5, ScalarType.Int16);
			one      = disperse.Generate();
			Print(one);
			one.dtype.Should().Be(ScalarType.Int16);

			disperse = new Discrete(5, ScalarType.Int32);
			one      = disperse.Generate();
			Print(one);
			one.dtype.Should().Be(ScalarType.Int32);

			disperse = new Discrete(5);
			one      = disperse.Generate();
			Print(one);
			one.dtype.Should().Be(ScalarType.Int64);
		}

		[Fact]
		public void DiscreteDevice()
		{
			var disperse = new Discrete(5, deviceType: DeviceType.CUDA);
			var one      = disperse.Generate();
			Print(one);
			one.device_type.Should().Be(DeviceType.CUDA);

			disperse = new Discrete(5, deviceType: DeviceType.CPU);
			one      = disperse.Generate();
			Print(one);
			one.device_type.Should().Be(DeviceType.CPU);
		}


		[Fact]
		public void DiscreteSample()
		{
			var a = new Discrete(5);
			foreach (var _ in Enumerable.Repeat(0, 10))
			{
				var data = a.Sample();
				Print(data);
				data.ToInt64().Should().BeInRange(0, 4);
			}
		}

		#endregion


		#region Box

		[Fact]
		public void CreateFloatBox()
		{
			var box = new Box(0f, 1f, new long[] { 2, 2 });
			var r   = box.Sample();
			Print(r);
		}

		[Fact]
		public void CreateDoubleBox()
		{
			var box = new Box(0d, 1d, new long[] { 2, 2 });
			var r   = box.Sample();
			Print(r);
		}


		[Fact]
		public void CreateInt32Box()
		{
			var box = new Box(1, 5, new long[] { 10 });
			var r   = box.Sample();
			Print(r);
		}

		[Fact]
		public void CreateInt64Box()
		{
			var box = new Box(1L, 5L, new long[] { 10 });
			var r   = box.Sample();
			Print(r);
		}

		[Fact]
		public void CreateByteBox()
		{
			var box = new Box((byte)0, (byte)1, new long[] { 10 });
			var r   = box.Sample();
			Print(r);
		}

		[Fact]
		public void CreateInt16Box()
		{
			var box = new Box((short)1, (short)5, new long[] { 10 });
			var r   = box.Sample();
			Print(r);
		}

		#endregion


		#region Other Space

		[Fact]
		public void CreateMultiDiscrete1()
		{
			var low           = tensor(new long[] { 0, 0 });
			var high          = tensor(new long[] { 3, 4 });
			var shape         = new long[] { 2 };
			var multiDiscrete = new MultiDiscrete(low, high, shape, ScalarType.Int32);
			var r             = multiDiscrete.Sample();
			Print(r);
			Print(multiDiscrete);
		}

		[Fact]
		public void CreateMultiDiscrete2()
		{
			var multiDiscrete = new MultiDiscrete(0, 1, new long[] { 2 }, ScalarType.Int64);
			var r             = multiDiscrete.Sample();
			Print(r);
			Print(multiDiscrete);
		}


		[Fact]
		public void CreateBinary()
		{
			var binary = new Binary(ScalarType.Int64);
			var r      = binary.Sample();
			Print(r);
			Print(binary);
		}


		[Fact]
		public void CreateMultiBinary1()
		{
			var binary = new MultiBinary(2L);
			var r      = binary.Sample();
			Print(r);
			Print(binary);
		}

		[Fact]
		public void CreateMultiBinary2()
		{
			var binary = new MultiBinary(new long[] { 2, 2 }, ScalarType.Int64);
			var r      = binary.Sample();
			Print(r);
			Print(binary);
		}

		#endregion
	}
}