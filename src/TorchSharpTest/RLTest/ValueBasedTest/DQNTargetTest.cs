using DeepSharp.RL.Agents.Deep.Value;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
	public class DQNTargetTest
	{
		[Fact]
		public void TerminalTransitionsDoNotBootstrap()
		{
			using var rewards    = tensor(new[] { 1f, 1f });
			using var nextValues = tensor(new[] { 10f, 10f });
			using var done       = tensor(new[] { true, false });
			using var targets    = DQN.CalculateTargets(rewards, nextValues, done, 0.5f);

			targets.data<float>().ToArray().Should().Equal(1f, 6f);
		}
	}
}