using RLSharp.Torch;
using RLSharp.Torch.Agents.Deep;
using RLSharp.Torch.Agents.Deep.Continuous;
using RLSharp.Torch.Agents.Deep.Policy;
using RLSharp.Torch.Environs;

namespace RLSharp.Tests.RLTest.PolicyBasedTest
{
	public class ModernPolicyAgentTest
	{
		[Fact]
		public void PpoLearnsOneBatchOnFrozenLake()
		{
			RandomProvider.SetSeed(42);
			var env   = new FrozenLake([0.8f, 0.1f, 0.1f]);
			var agent = new PPO(env, 1, updateEpochs: 1);

			var outcome = agent.Learn();

			outcome.Steps.Should().NotBeEmpty();
			float.IsFinite(outcome.Evaluate).Should().BeTrue();
		}

		[Theory]
		[InlineData("DDPG")]
		[InlineData("TD3")]
		[InlineData("SAC")]
		public void ContinuousControlAgentsRequireContinuousEnvironmentAdapter(string algorithm)
		{
			var env = new FrozenLake([0.8f, 0.1f, 0.1f]);
			DeepAgent agent = algorithm switch
			{
				"DDPG" => new DDPG(env),
				"TD3"  => new TD3(env),
				"SAC"  => new SAC(env),
				_      => throw new ArgumentOutOfRangeException(nameof(algorithm))
			};

			var learn = () => agent.Learn();

			learn.Should()
				.Throw<NotSupportedException>()
				.WithMessage("*continuous-action environment adapter*");
		}
	}
}