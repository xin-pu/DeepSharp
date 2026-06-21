namespace TorchSharpTest.TorchTests
{
	public class TensorTest : AbstractTest
	{
		public TensorTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		[Fact]
		public void CreateRandTensor()
		{
			var device = new Device(DeviceType.CUDA);
			var tensor = randn(3, 5, 4, device: device);
			Print(tensor.ToString());
		}

		[Fact]
		public void CreateArrayTensor()
		{
			var tensor = from_array(new float[] { 1, 2 }).to(DeviceType.CUDA);
			Print(tensor);
		}

		[Fact]
		public void CreateOnesTensor()
		{
			var tensor = ones(2, 3).to(DeviceType.CUDA);
			Print(tensor);
		}


		[Fact]
		public void TestAnyAndAll()
		{
			var a = tensor(new long[] { 1, 2, 3 });
			var b = tensor(new long[] { 0, 0, 0 });
			all(b < a).Equals(tensor(true)).Should().Be(true);
		}


		[Fact]
		public void TestM()
		{
			var probs  = tensor(new[] { 1 / 3f, 1 / 3f, 1 / 3f });
			var sample = multinomial(probs, 1000, true);

			var arr = sample.data<long>();
			var a1  = arr.Count(a => a == 0);
			var a2  = arr.Count(a => a == 1);
			var a3  = arr.Count(a => a == 2);
			Print($"{a1},{a2},{a3}");
		}


		[Fact]
		public void TestGreatAndLess()
		{
			var input       = tensor(new[] { 1.1f, 3f, 3.3f, 5.2f });
			var targetLower = 2f;
			var targetUpper = 4f;

			var mean = (targetUpper + targetLower) / 2;
			var half = (targetUpper - targetLower) / 2;
			var dis  = 1 - abs(input - mean) / half;
			Print(dis);

			var condition = dis.greater(0);
			Print(condition);

			var final = dis.where(condition, 0);
			Print(final);
		}
	}
}