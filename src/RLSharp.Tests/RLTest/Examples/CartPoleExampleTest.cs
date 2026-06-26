using RLSharp.Torch.Agents.Generic;
using RLSharp.Torch.Encoding;
using RLSharp.Torch.Examples.CartPole;

namespace RLSharp.Tests.RLTest.Examples
{
	public class CartPoleExampleTest
	{
		[Fact]
		public void CartPoleResetAndStepProduceValidState()
		{
			var env = new CartPoleEnvironment(42, 10);

			var initial = env.Reset();
			var result  = env.Step(CartPoleAction.Right);

			initial.ToFeatures().Should().HaveCount(4);
			result.State.ToFeatures().Should().HaveCount(4);
			result.Reward.Should().BeGreaterThanOrEqualTo(0f);
			result.Reward.Should().BeLessThanOrEqualTo(1f);
		}

		[Fact]
		public void DqnAgentCanLearnOneCartPoleEpisode()
		{
			var env     = new CartPoleEnvironment(42, 20);
			var encoder = new DelegateStateEncoder<CartPoleState>(4, state => state.ToFeatures());
			var agent = new DqnAgent<CartPoleState, CartPoleAction>(
				env,
				encoder,
				epsilon: 0.2f,
				maxStepsPerEpisode: 20,
				seed: 42);

			var result = agent.Learn();

			result.Steps.Should().BeGreaterThan(0);
			result.Reward.Should().BeGreaterThanOrEqualTo(0f);
			result.Loss.Should().NotBeNull();
			float.IsFinite(result.Loss!.Value).Should().BeTrue();
		}

		[Fact]
		public void PpoAgentCanLearnOneCartPoleEpisode()
		{
			var env     = new CartPoleEnvironment(42, 20);
			var encoder = new DelegateStateEncoder<CartPoleState>(4, state => state.ToFeatures());
			var agent = new PpoAgent<CartPoleState, CartPoleAction>(
				env,
				encoder,
				maxStepsPerEpisode: 20);

			var result = agent.Learn();

			result.Steps.Should().BeGreaterThan(0);
			result.Reward.Should().BeGreaterThanOrEqualTo(0f);
			result.Loss.Should().NotBeNull();
			float.IsFinite(result.Loss!.Value).Should().BeTrue();
		}
	}
}