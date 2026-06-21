namespace TorchSharpTest.LossTest
{
	public class LossTest : AbstractTest
	{
		public LossTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		[Fact]
		public void Cross()
		{
			var input  = randn(3, 5, requires_grad: true);
			var target = empty(3, ScalarType.Int64).randint_like(0, 5);
			var loss   = CrossEntropyLoss();
			var c      = loss.call(input, target);
			c.backward();
			var array = c.data<float>().ToArray();
			Print(string.Join(",", array));
		}
	}
}