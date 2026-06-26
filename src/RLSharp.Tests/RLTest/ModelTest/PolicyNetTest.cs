using RLSharp.Torch.Agents;

namespace RLSharp.Tests.RLTest.ModelTest
{
	public class PolicyNetTest : AbstractTest
	{
		public PolicyNetTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		[Fact]
		public void TestNet()
		{
			var net = new PGN(3, 128, 2);
			var res = net.forward(from_array(new float[,] { { 1, 2, 3 } }));
			Print(res);
		}
	}
}