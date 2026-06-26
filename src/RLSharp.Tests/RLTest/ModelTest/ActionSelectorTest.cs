using RLSharp.Torch.ActionSelectors;

namespace RLSharp.Tests.RLTest.ModelTest
{
	public class ActionSelectorTest : AbstractTest
	{
		public ActionSelectorTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		[Fact]
		public void ArgmaxActionSelectorTest()
		{
			var input = from_array(new double[,] { { 1, 2, 3 }, { 1, -1, 0 } });
			var res   = new ArgmaxActionSelector().Select(input);
			res.Equals(tensor(new long[] { 2, 0 })).Should().BeTrue();
		}

		[Fact]
		public void ProbabilityActionSelectorTest()
		{
			var input = from_array(new[,] { { 1f, 0, 0 }, { 0, 1f, 0 } });
			var res   = new ProbActionSelector().Select(input);
			res.Equals(from_array(new long[] { 0, 1 })).Should().BeTrue();
		}

		[Fact]
		public void EpsilonActionSelectorUsesGreedyActionsWhenEpsilonIsZero()
		{
			using var input  = from_array(new[,] { { 1f, 3f, 2f }, { 4f, 1f, 0f } });
			using var result = new EpsilonActionSelector(new ArgmaxActionSelector(), 0).Select(input);

			result.data<long>().ToArray().Should().Equal(1, 0);
		}

		[Fact]
		public void EpsilonActionSelectorValidatesEpsilon()
		{
			var create = () => new EpsilonActionSelector(new ArgmaxActionSelector(), 1.1f);
			create.Should().Throw<ArgumentOutOfRangeException>();
		}
	}
}