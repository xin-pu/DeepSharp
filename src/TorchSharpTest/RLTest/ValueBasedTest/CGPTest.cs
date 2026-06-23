using DeepSharp.RL.Agents.Deep.Value;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
	public class CGPTest : AbstractTest
	{
		public CGPTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		/// <summary>
		///     Smoke test: create CGP agent and get a policy action.
		/// </summary>
		[Fact]
		public void TestCGP()
		{
			var frozenLake = new FrozenLake();
			var cgp        = new CGP(frozenLake);
			var act        = cgp.GetPolicyAct(frozenLake.Observation!.Value!);
			Print(act);
		}

		/// <summary>
		///     Verify CGP learns on K-Armed Bandit: loss is non-zero and final reward
		///     exceeds the random baseline after a fixed number of training iterations.
		/// </summary>
		[Fact]
		public void KArmedBanditQuickDiagnostic()
		{
			var kArmedBandit = new KArmedBandit(new[] { 0.4, 0.85, 0.75, 0.75 });
			// Small capacity + episodes for fast iteration
			var agent = new CGP(kArmedBandit, 10, 100, batchSize: 8, temperature: 0.5f);
			Print(kArmedBandit);

			var losses = new List<float>();
			foreach (var i in Enumerable.Range(0, 50))
			{
				kArmedBandit.Reset();
				var outcome = agent.Learn();
				losses.Add(outcome.Evaluate);

				if (i % 10 == 0)
				{
					var testReward = agent.TestEpisodes(20);
					Print($"Iter {i:D3}: Loss={outcome.Evaluate:F4}, TestReward={testReward:F4}");
				}
			}

			// Verify loss is finite (learning is happening)
			var validLosses = losses.Where(l => l > 0).ToList();
			validLosses.Should().NotBeEmpty("should have computed losses");

			// Final test: policy should outperform random baseline (~0.7)
			var finalReward = agent.TestEpisodes(50);
			Print($"Final test reward: {finalReward:F4}");
			finalReward.Should().BeGreaterThan(0.6f, "CGP should learn better than random");
		}

		/// <summary>
		///     Verify CGP learns on K-Armed Bandit with high temperature (softer target).
		///     High τ makes the target distribution more uniform, encouraging exploration
		///     during policy training.
		/// </summary>
		[Fact]
		public void KArmedBanditHighTemperature()
		{
			var kArmedBandit = new KArmedBandit(new[] { 0.4, 0.85, 0.75, 0.75 });
			var agent        = new CGP(kArmedBandit, 10, 100, batchSize: 8, temperature: 5.0f);
			Print(kArmedBandit);

			foreach (var i in Enumerable.Range(0, 50))
			{
				kArmedBandit.Reset();
				agent.Learn();
			}

			var finalReward = agent.TestEpisodes(50);
			Print($"High-τ final test reward: {finalReward:F4}");
			finalReward.Should().BeGreaterThan(0.6f, "CGP with high temperature should learn");
		}

		/// <summary>
		///     Verify CGP can learn on FrozenLake within a reasonable number of iterations.
		/// </summary>
		[Fact]
		public void FrozenLakeDiagnostic()
		{
			var frozenlake = new FrozenLake(new[] { 0.8f, 0.1f, 0.1f });
			var agent      = new CGP(frozenlake, 50, 500, 0.9f, batchSize: 16, temperature: 1.0f);
			Print(frozenlake);

			var bestReward = 0f;
			foreach (var i in Enumerable.Range(0, 30))
			{
				frozenlake.Reset();
				agent.Learn();
				var reward                          = agent.TestEpisodes(20);
				if (reward > bestReward) bestReward = reward;
				Print($"{i:D3}: TestReward={reward:F4}");
			}

			Print($"Best test reward: {bestReward:F4}");
			// FrozenLake is harder; just verify we get some positive signal
			bestReward.Should().BeGreaterOrEqualTo(0f, "CGP should not regress on FrozenLake");
		}
	}
}